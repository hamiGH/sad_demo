# Speech Activity Detection (SAD) service

Massege format: A FormData Object including two fields, first one is ‘file’ which its value is an audio data (mp3 or wav) and the second is ‘threshold’ which ia a folat number between 0 and 1.

Output format: A json file which has one field named ‘sad_annotation’ which is a list of begin and end times (in seconds) of all speechs within the audio file.

Output example:
{
 "sad_annotation": [
             { 
              "begin": 0.76,
              "end": 1.06 
             }, 
             {
               "begin": 1.66,
               "end": 2.36 
              } 
             ] 
}

limits and constraints: audio chunks should not exceed 10mb.

Relevent tags: #SAD #sad #speech_activity_detection #voice_activity_detection #speech_detection #voice_detection #detect_speech #speech_analysis #voice_analysis #speech_processing #voice_processing #AI #ai 


## Acknowledgements

This project includes code from the [OriginalRepository](https://github.com/original-repo) by [OriginalAuthor](https://github.com/original-author). The following files have been adapted or used from the original repository:

- [File1](path/to/File1): Description of the modifications made.
- [File2](path/to/File2): Description of the modifications made.
