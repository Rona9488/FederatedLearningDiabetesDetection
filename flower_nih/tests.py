import pytest
import torch
import pandas as pd
from flower_nih.task import Net, load_data, train, test, predict_single

# Dummy dataset untuk test
@pytest.fixture
def dummy_dataframe():
    # 100 sampel, 10 fitur, 3 kelas (0, 1, 2)
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    df = pd.DataFrame(torch.cat([X, y.unsqueeze(1)], dim=1).numpy())
    return df

def test_model_forward_pass():
    model = Net(input_size=10, hidden_size=16, num_classes=3)
    x = torch.randn(4, 10)
    output = model(x)
    assert output.shape == (4, 3), "Output shape should be (batch_size, num_classes)"

def test_data_loading(dummy_dataframe):
    train_loader, test_loader = load_data(batch_size=8, df=dummy_dataframe)
    assert len(train_loader) > 0
    assert len(test_loader) > 0
    for x, y in train_loader:
        assert x.shape[1] == 10
        assert len(x) == len(y)
        break

def test_training_loop(dummy_dataframe):
    device = torch.device("cpu")
    model = Net(input_size=10, hidden_size=16, num_classes=3)
    train_loader, test_loader = load_data(batch_size=8, df=dummy_dataframe)
    results = train(model, train_loader, test_loader, epochs=1, learning_rate=0.001, device=device)
    assert "val_loss" in results
    assert "val_accuracy" in results
    assert 0 <= results["val_accuracy"] <= 1

def test_model_prediction():
    model = Net(input_size=10, hidden_size=16, num_classes=3)
    model.eval()
    dummy_input = torch.randn(10).tolist()
    pred = predict_single(model, dummy_input)
    assert isinstance(pred, int)
    assert 0 <= pred < 3
