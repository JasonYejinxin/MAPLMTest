Traceback (most recent call last):
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 146, in <module>
    train_model(qa_data, feature_method="concatenate")  # 或者改为 "concatenate"
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/airlab/Desktop/Jingwen/MAPLMTest/baseline/evaluation/data_processed_concat.py", line 127, in train_model
    loss.backward()
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/_tensor.py", line 581, in backward
    torch.autograd.backward(
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/home/airlab/anaconda3/lib/python3.12/site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
