# Classification-of-Dialogue-Segment-Breaks---CSE582
This is the repository of Group 4 for CSE582 Final Project.


## Folder to upload Preprocessed Datasets - [Click Here](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/hmp5565_psu_edu/EuAqhN2dA5ZAg3xCPeRZwNsB4pZquDW6onc20c9BGTo-XQ?e=hYBv6N)

### Variable Descriptions: 
- **category_name**: The name of the activity.

| category_name      | Frequency | Percentage |
|--------------------|-----------|------------|
| using_microwave    | 768       | 15.06%     |
| wash_glass         | 655       | 12.84%     |
| using_oven         | 649       | 12.73%     |
| using_computer     | 640       | 12.55%     |
| bedroom_cabinet    | 626       | 12.27%     |
| watch_tv           | 621       | 12.18%     |
| using_stove        | 606       | 11.88%     |
| read_book          | 535       | 10.49%     |

- **category_number**: The activity variants, or confusion types. Each activity type has four variant videos, indexed from 0 to 3, where 0 is a variant where the alien is able to complete the task with no confusion; 1 represents a confusion that causes the alien to freeze for a moment due to cognitive overload; 2 represents the alien using the wrong object; 3 represents a wrong location.

| category_number | Frequency | Percentage |
|-----------------|-----------|------------|
| 0               | 1226      | 24.04%     |
| 1               | 1188      | 23.29%     |
| 2               | 1472      | 28.86%     |
| 3               | 1214      | 23.80%     |

- **change_speaker**: Whether or not the two utterances happen between two different users/speakers. 1 indicates two different speakers, 0 indicates the same speaker.

| change_speaker | Frequency | Percentage |
|----------------|-----------|------------|
| 0              | 1091      | 21.39%     |
| 1              | 4009      | 78.61%     |

- **utterance1_intent**: The intent label for utterance 1.
- **utterance2_intent**: The intent label for utterance 2. 
