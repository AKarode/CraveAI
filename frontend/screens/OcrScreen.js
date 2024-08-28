import React, { useState } from 'react';
import { View, Text, Button, Image, StyleSheet, Alert, ScrollView, ImageBackground } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import backgroundImage from '../assets/background.png'; // Import the background image

const OcrScreen = () => {
  const [selectedImages, setSelectedImages] = useState([]);

  const pickImagesFromCamera = async () => {
    let permissionResult = await ImagePicker.requestCameraPermissionsAsync();

    if (permissionResult.granted === false) {
      Alert.alert('Permission to access camera is required!');
      return;
    }

    let results = [];
    const options = {
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
      base64: true, // Include base64 to avoid issues with large images
    };

    // Loop to allow multiple images
    for (let i = 0; i < 3; i++) { // Change the number if you want to limit the number of images
      let result = await ImagePicker.launchCameraAsync(options);

      if (!result.canceled) {
        results.push(result.assets[0]);
      } else {
        break;
      }
    }

    setSelectedImages(prevImages => [...prevImages, ...results]);
  };

  const pickImagesFromGallery = async () => {
    let permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      Alert.alert('Permission to access media library is required!');
      return;
    }

    let result = await ImagePicker.launchImageLibraryAsync({
      allowsMultipleSelection: true,
      selectionLimit: 10, // Change this number to limit the number of images
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
    });

    if (!result.canceled) {
      setSelectedImages(prevImages => [...prevImages, ...result.assets]);
    }
  };

  const sendImagesToServer = async () => {
    if (selectedImages.length === 0) {
      Alert.alert('No images selected', 'Please select images before sending.');
      return;
    }

    const formData = new FormData();
    selectedImages.forEach((image, index) => {
      formData.append('images[]', {
        uri: image.uri,
        name: `photo${index}.jpg`,
        type: 'image/jpeg',
      });
    });

    try {
      let response = await fetch('http://localhost:5000/endpoint', {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData,
      });

      let data = await response.json();
      if (response.ok) {
        Alert.alert('Success', 'Images sent successfully');
      } else {
        Alert.alert('Error', data.error || 'Failed to send the images');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to send images to the server');
      console.error('Error sending images:', error);
    }
  };

  return (
    <ImageBackground source={backgroundImage} style={styles.background}>
      <View style={styles.container}>
        <Text style={styles.text}>OCR Screen</Text>
        <Button title="Take Pictures" onPress={pickImagesFromCamera} />
        <Button title="Pick from Camera Roll" onPress={pickImagesFromGallery} />
        {selectedImages.length > 0 && (
          <ScrollView contentContainerStyle={styles.imageContainer}>
            {selectedImages.map((image, index) => (
              <Image
                key={index}
                source={{ uri: image.uri }}
                style={styles.image}
              />
            ))}
            <Button title="Send Images to Server" onPress={sendImagesToServer} />
          </ScrollView>
        )}
      </View>
    </ImageBackground>
  );
};

const styles = StyleSheet.create({
  background: {
    flex: 1,
    resizeMode: 'cover',
    justifyContent: 'center',
  },
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 60,
    color: '#000000',
  },
  imageContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginTop: "100"
  },
  image: {
    width: 100,
    height: 100,
    resizeMode: 'contain',
    margin: 5,
  },
});

export default OcrScreen;
