Error occurred: Given groups=1, weight of size [768, 3, 16, 16], expected input[1, 12, 224, 224] to have 3 channels, but got 12 channels instead
input_ids shape: torch.Size([1, 9])
pixel_values shape: torch.Size([1, 12, 224, 224])
labels shape: torch.Size([1, 9])
Error occurred: Given groups=1, weight of size [768, 3, 16, 16], expected input[1, 12, 224, 224] to have 3 channels, but got 12 channels instead
input_ids shape: torch.Size([1, 19])
pixel_values shape: torch.Size([1, 12, 224, 224])
labels shape: torch.Size([1, 19])
Epoch1 completed, but loss was not coumputed
Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 145, in <module>
    train_model(qa_data, feature_method="concatenate")  # 或者改为 "concatenate"
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 135, in train_model
    print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")
                                                ^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'item'
