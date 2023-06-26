# Speech Activity Detection (SAD) service

**Network Protocol:** HTTP

**Server Framework:** Flask

**Endpoint:** `url:port:/sad`

**Message Format:**

A `FormData` object including two fields:

1. `file`: An audio data file (MP3 or WAV).
2. `threshold`: A floating-point number between 0 and 1.

**Output Format:**

A JSON file with a field named `sad_annotation`, which is a list of speech segments within the audio file. Each segment contains `begin` and `end` fields indicating the start and end times of the speech segment in seconds.

**Output Example:**

```json
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
```

<!--- Relevent tags: #SAD #sad #speech_activity_detection #voice_activity_detection #speech_detection #voice_detection #detect_speech #speech_analysis #voice_analysis #speech_processing #voice_processing #AI #ai -->

**Limits and Constraints**:
- Audio chunks should not exceed 10MB.

## Launching the Server:
To launch the server using Docker and docker-compose, run the following command in the root directory of project where the docker-compose.yml file is located:

```sh
sudo docker-compose up
```

This command will start the server container based on the configuration in the docker-compose.yml file.
You can now access the server at http://localhost:5005/sad or the appropriate URL and port based on your configuration.

Make sure you have Docker and docker-compose installed and properly configured on your machine before running these commands.

Feel free to adjust the configuration in the Dockerfile and docker-compose.yml file based on your specific server setup and requirements.


## Acknowledgements

This project includes code from the [SincNet](https://github.com/mravanelli/SincNet) by [mravanelli](https://github.com/mravanelli). The following file have been adapted or used from the original repository:

- [dnn_models](https://github.com/hamiGH/sad_demo/blob/main/dnn_models.py)