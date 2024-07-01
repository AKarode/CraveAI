import React from 'react';
import { View, Text, StyleSheet, ImageBackground, Image, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import backgroundImage from '../assets/background.png';
import middleImage from '../assets/food.jpg'; // Replace this with your actual image

export default function HomeScreen() {
  const navigation = useNavigation();

  const navigateToChat = () => {
    navigation.navigate('ChatScreen');
  };

  return (
    <View style={styles.container}>
      <ImageBackground source={backgroundImage} style={styles.image}>
        <Text style={styles.title}>CraveAI</Text>
        <View style={styles.middleContainer}>
          <Image source={middleImage} style={styles.middleImage} />
        </View>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={navigateToChat}>
            <Text style={styles.buttonText}>CHATBOT</Text>
          </TouchableOpacity>
        </View>
      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000000',
  },
  image: {
    flex: 1,
    resizeMode: 'cover',
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000000',
  },
  title: {
    fontSize: 48,
    fontWeight: 'bold',
    color: 'white',
    marginTop: 80,
  },
  middleContainer: {
    flex: 1,
    marginTop: 90,
    alignItems: 'center', // Center items horizontally
  },
  middleImage: {
    width: 400,
    height: 250,
  },
  buttonContainer: {
    marginBottom: 50, // Adjust as needed for spacing
    alignItems: 'center', // Center button horizontally
  },
  button: {
    backgroundColor: '#FFFFFF',
    paddingVertical: 10,
    paddingHorizontal: 40,
    borderRadius: 20,
  },
  buttonText: {
    fontSize: 24,
    color: 'black',
    fontWeight: 'bold',
  },
});

