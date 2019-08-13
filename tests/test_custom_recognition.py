from imageai.Prediction.Custom import CustomImagePrediction
import os
import pytest
from os.path import dirname


TEST_FOLDER = os.path.dirname(__file__)

all_images = os.listdir(os.path.join(TEST_FOLDER, "data-images"))
all_images_array = []


def images_to_image_array():
    for image in all_images:
        all_images_array.append(os.path.join(TEST_FOLDER, "data-images", image))



@pytest.mark.resnet
@pytest.mark.recognition_custom
def mytest_custom_recognition_model_resnet():
    predictor = CustomImagePrediction()
    predictor.setModelTypeAsResNet()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "idenprof_resnet.h5"))
    predictor.setJsonPath(model_json=os.path.join(TEST_FOLDER, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(TEST_FOLDER, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], str)


@pytest.mark.resnet
@pytest.mark.recognition_custom
def mytest_custom_recognition_full_model_resnet():
    predictor = CustomImagePrediction()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "idenprof_full_resnet_ex-001_acc-0.119792.h5"))
    predictor.setJsonPath(model_json=os.path.join(TEST_FOLDER, "data-json", "idenprof.json"))
    predictor.loadFullModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(TEST_FOLDER, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], str)


@pytest.mark.densenet
@pytest.mark.recognition_custom
def mytest_custom_recognition_model_densenet():
    predictor = CustomImagePrediction()
    predictor.setModelTypeAsDenseNet()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "idenprof_densenet-0.763500.h5"))
    predictor.setJsonPath(model_json=os.path.join(TEST_FOLDER, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    predictions, probabilities = predictor.predictImage(image_input=os.path.join(TEST_FOLDER, "data-images", "9.jpg"))

    assert isinstance(predictions, list)
    assert isinstance(probabilities, list)
    assert isinstance(predictions[0], str)
    assert isinstance(probabilities[0], str)




@pytest.mark.resnet
@pytest.mark.recognition_custom
@pytest.mark.recognition_multi
def mytest_custom_recognition_model_resnet_multi():
    predictor = CustomImagePrediction()
    predictor.setModelTypeAsResNet()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "idenprof_resnet.h5"))
    predictor.setJsonPath(model_json=os.path.join(TEST_FOLDER, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], str)


@pytest.mark.resnet
@pytest.mark.recognition_custom
@pytest.mark.recognition_multi
def mytest_custom_recognition_full_model_resnet_multi():
    predictor = CustomImagePrediction()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "idenprof_full_resnet_ex-001_acc-0.119792.h5"))
    predictor.setJsonPath(model_json=os.path.join(TEST_FOLDER, "data-json", "idenprof.json"))
    predictor.loadFullModel(num_objects=10)
    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], str)


@pytest.mark.densenet
@pytest.mark.recognition_custom
@pytest.mark.recognition_multi
def mytest_custom_recognition_model_densenet_multi():
    predictor = CustomImagePrediction()
    predictor.setModelTypeAsDenseNet()
    predictor.setModelPath(os.path.join(TEST_FOLDER, "data-models", "idenprof_densenet-0.763500.h5"))
    predictor.setJsonPath(model_json=os.path.join(TEST_FOLDER, "data-json", "idenprof.json"))
    predictor.loadModel(num_objects=10)
    images_to_image_array()
    result_array = predictor.predictMultipleImages(sent_images_array=all_images_array)

    assert isinstance(result_array, list)
    for result in result_array:
        assert "predictions" in result
        assert "percentage_probabilities" in result
        assert isinstance(result["predictions"], list)
        assert isinstance(result["percentage_probabilities"], list)
        assert isinstance(result["predictions"][0], str)
        assert isinstance(result["percentage_probabilities"][0], str)