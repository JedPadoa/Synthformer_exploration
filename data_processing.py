import numpy as np
import pandas as pd
import os
import soundfile as sf
import dac
from audiotools import AudioSignal
import shutil
from sklearn.model_selection import train_test_split
import soundfile as sf
from pathlib import Path
import librosa

DATA_DIR = 'ESC-50-master/audio/'
FILTERED_DATA_DIR = 'ESC-50-master/non_silent_audio'
METADATA = 'ESC-50-master/meta/esc50.csv'
SUBSET_DIR = 'data_subset'
DAC_DIR = 'DAC_FILES'
classes = ['clicky', 'crackly', 'swooshy', 'hummy']

classes_map = {
    'clicky': (['mouse_click', 'clock_tick', 'keyboard_typing', 'door_wood_knock', 'footsteps'], 0),
    'crackly': (['fireworks', 'crackling_fire', 'glass_breaking', 'clapping'], 0.33),
    'swooshy': (['vacuum_cleaner', 'thunderstorm', 'airplane', 'pouring_water', 'washing_machine', 'sea_waves'], 0.99),
    'hummy': (['airplane', 'engine', 'helicopter'], 0.66),
    }

classes_list = ['dog' 'chirping_birds' 'vacuum_cleaner' 'thunderstorm' 'door_wood_knock'
                'can_opening' 'crow' 'clapping' 'fireworks' 'chainsaw' 'airplane'
                'mouse_click' 'pouring_water' 'train' 'sheep' 'water_drops'
                'church_bells' 'clock_alarm' 'keyboard_typing' 'wind' 'footsteps' 'frog'
                'cow' 'brushing_teeth' 'car_horn' 'crackling_fire' 'helicopter'
                'drinking_sipping' 'rain' 'insects' 'laughing' 'hen' 'engine' 'breathing'
                'crying_baby' 'hand_saw' 'coughing' 'glass_breaking' 'snoring'
                'toilet_flush' 'pig' 'washing_machine' 'clock_tick' 'sneezing' 'rooster'
                'sea_waves' 'siren' 'cat' 'door_wood_creaks' 'crickets']

class Preprocessor:
    def __init__(self, data_dir, metadata, DAC_DIR, subset_dir, classes):
        self.data_dir = data_dir
        self.metadata = metadata
        self.classes = classes
        self.original_df = self._process_df(pd.read_csv(self.metadata))
        self.dac_dir = DAC_DIR
        self.subset_dir = subset_dir
    
    def split_train_test(self):
        # Get all DAC files
        dac_files = [f.replace('.dac', '.wav') for f in os.listdir(self.dac_dir) if f.endswith('.dac')]
        df = self.original_df[self.original_df['Full File Name'].isin(dac_files)]
        y = df['Class Name'].values
        # Split files into train and test sets (80-20 split)
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=y, random_state=42)
        # Replace .wav with .dac in the Full File Name column
        self._rename_wav_to_dac(train_df)
        self._rename_wav_to_dac(test_df)
        print(train_df.head(), test_df.head())

        # Create train and test directories
        train_dir = os.path.join('processed_data', 'train')
        test_dir = os.path.join('processed_data', 'test')

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        train_df.to_excel('processed_data/dac-train.xlsx', index=False)
        test_df.to_excel('processed_data/dac-test.xlsx', index=False)

        # Move files to respective directories
        for file in train_df['Full File Name'].values:
            src_path = os.path.join(DAC_DIR, file)
            dest_path = os.path.join(train_dir, file)
            shutil.copy(src_path, dest_path)

        for file in test_df['Full File Name'].values:
            src_path = os.path.join(DAC_DIR, file)
            dest_path = os.path.join(test_dir, file)
            shutil.copy(src_path, dest_path)
            
        train_classes = train_df['Class Name'].unique()
        test_classes = test_df['Class Name'].unique()

        print(f"Training set size: {len(train_df)} files")
        print(f"Test set size: {len(test_df)} files")
        print(f"Training set has {len(train_classes)} classes.")
        print(f"Test set has {len(test_classes)} classes.")
    
    def get_subset(self):
        os.makedirs(self.subset_dir, exist_ok=True)
        files = [f for f in self.original_df['Full File Name'].values]
        for f in files:
            if f in os.listdir(self.data_dir):
                shutil.copy(os.path.join(self.data_dir, f), os.path.join(self.subset_dir, f))
        print(f'Found {len(os.listdir(self.subset_dir))} files matching the required classes')
        return None
     
    def get_non_silent_files(self, data_dir): 
        os.makedirs(self.filtered_data_dir, exist_ok=True)
        for f in os.listdir(data_dir):
            file = os.path.join(data_dir, f)
            if self._silence_dur(file) < 0.5:
                shutil.copy(os.path.join(data_dir, f), os.path.join(self.filtered_data_dir, f))
        return self.filtered_data_dir
       
    def _process_df(self, df):
        df.drop(['fold', 'target', 'esc10', 'src_file', 'take'], axis=1, inplace=True)
        df.rename({'category': 'Class Name'}, axis=1, inplace=True)
        df.rename({'filename': 'Full File Name'}, axis=1, inplace=True)
        df['Param1'] = 1
        print(f"Shape of DataFrame: {df.shape}")
        print(df.head())
        return self._filter_df(self._map_classes(df, classes_map))
    
    def _rename_wav_to_dac(self, df):
        for index, row in df.iterrows():
            filename = row['Full File Name']
            new_filename = filename.replace('.wav', '.dac')
            df.at[index, 'Full File Name'] = new_filename
        return df

    def _get_unique(self, df):
        return df['Class Name'].unique()
    
    def _filter_df(self, df): 
        return df[df['Class Name'].isin(self.classes)]
    
    def _map_classes(self, df, classes_map):
        ## map class names to their corrresponding keys in the classes_map
        for index, row in df.iterrows():
            original_class = row['Class Name']
            # Find which category the class belongs to
            for category, classes in classes_map.items():
                if original_class in classes[0]:
                    df.at[index, 'Class Name'] = category
                    df.at[index, 'Param1'] = classes[1]
                    break      
        return df
    
def silence_dur(file):
        y, sr = librosa.load(file, sr=None)
        non_silent_intervals = librosa.effects.split(y, top_db=25, frame_length=2048)
        # Calculate the total duration of non-silent intervals
        non_silent_duration = np.sum([(end - start) for start, end in non_silent_intervals]) / sr
        # Calculate the total duration of the audio
        total_duration = len(y) / sr
        # Calculate the total duration of silence
        silence_duration = total_duration - non_silent_duration
        return silence_duration 
    
def get_non_silent_files(data_dir): 
        os.makedirs(FILTERED_DATA_DIR, exist_ok=True)
        for f in os.listdir(data_dir):
            file = os.path.join(data_dir, f)
            if silence_dur(file) < 0.5:
                shutil.copy(os.path.join(data_dir, f), os.path.join(FILTERED_DATA_DIR, f))
        print(f"Number of non-silent files: {len(os.listdir(FILTERED_DATA_DIR))}")
    
if __name__ == '__main__':
    get_non_silent_files(DATA_DIR)
    preprocessor = Preprocessor(FILTERED_DATA_DIR, METADATA, DAC_DIR, SUBSET_DIR, classes)
    preprocessor.get_subset()
    #encode to dac in folder DAC_FILES before doing this
    preprocessor.split_train_test()
    
    