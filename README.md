input_ids shape: torch.Size([1, 16])
pixel_values shape: torch.Size([4, 3, 224, 224])
labels shape: torch.Size([1, 16])
Error occurred: Expected input batch_size (32) to match target batch_size (8).
input_ids shape: torch.Size([1, 9])
pixel_values shape: torch.Size([4, 3, 224, 224])
labels shape: torch.Size([1, 9])
Error occurred: Expected input batch_size (72) to match target batch_size (18).
input_ids shape: torch.Size([1, 19])
pixel_values shape: torch.Size([4, 3, 224, 224])
labels shape: torch.Size([1, 19])
Epoch1 completed, but loss was not coumputed
Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 144, in <module>
    train_model(qa_data, feature_method="concatenate")  # 或者改为 "concatenate"
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 134, in train_model
    print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")
                                                ^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'item'
