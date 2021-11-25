from PIL import Image, ImageOps
from numpy import asarray, array, concatenate, expand_dims
from os import listdir
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from Googlenet import GoogLeNet


image_shape = (224, 224)
y_values = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'add': 10, 'sub': 11,
            'eq': 12}


def read_folder(location, image_size, category):
    folder_list = list()
    y_axis = list()

    for file in listdir(location):
        image = Image.open(location + '/' + file).convert('RGB')
        image = image.resize(image_size)
        image = ImageOps.grayscale(image)
        img_array = asarray(image)
        img_array = expand_dims(img_array, axis=2)
        folder_list.append(img_array)
        inverted_image = ImageOps.invert(image)
        inverted_array = asarray(inverted_image)
        img_array = expand_dims(img_array, axis=2)
        folder_list.append(inverted_array)

    folder_array = array(folder_list, dtype=list)

    for i in listdir(location):
        y = y_values[category]
        y2 = y
        y_axis.append(y)
        y_axis.append(y2)

    y_axis = array(y_axis, dtype=list)

    return folder_array, y_axis

location = 'Dataset/'

if __name__ == '__main__':
    x_array_0, y_array_0 = read_folder('/Dataset/0', image_shape, '0')
    x_array_1, y_array_1 = read_folder('Dataset/1', image_shape, '1')
    x_array_2, y_array_2 = read_folder('Dataset/2', image_shape, '2')
    x_array_3, y_array_3 = read_folder('Dataset/3', image_shape, '3')
    x_array_4, y_array_4 = read_folder('Dataset/4', image_shape, '4')
    x_array_5, y_array_5 = read_folder('Dataset/5', image_shape, '5')
    x_array_6, y_array_6 = read_folder('Dataset/6', image_shape, '6')
    x_array_7, y_array_7 = read_folder('Dataset/7', image_shape, '7')
    x_array_8, y_array_8 = read_folder('Dataset/8', image_shape, '8')
    x_array_9, y_array_9 = read_folder('Dataset/9', image_shape, '9')
    x_array_add, y_array_add = read_folder('Dataset/add', image_shape, 'add')
    x_array_sub, y_array_sub = read_folder('Dataset/sub', image_shape, 'sub')
    x_array_eq, y_array_eq = read_folder('Dataset/eq', image_shape, 'eq')

    x = concatenate((x_array_0, x_array_1, x_array_2, x_array_3, x_array_4, x_array_5, x_array_6, x_array_7, x_array_8,
                     x_array_9, x_array_add, x_array_sub, x_array_eq))
    y = concatenate((y_array_0, y_array_1, y_array_2, y_array_3, y_array_4, y_array_5, y_array_6, y_array_7, y_array_8,
                     y_array_9, y_array_add, y_array_sub, y_array_eq))

    x, y = shuffle(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train = asarray(x_train).astype('float32')
    y_train = asarray(y_train).astype('float32')
    x_test = asarray(x_test).astype('float32')
    y_test = asarray(y_test).astype('float32')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    output_shape = y_train.shape[1]
    model = GoogLeNet(output_shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=9, batch_size=40, validation_data=(x_test, y_test))
    model.save('D:/PycharmProjects/Automatic-calculator/Saved_Models/0-eq.h5')
