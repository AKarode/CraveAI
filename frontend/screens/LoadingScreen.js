import React from 'react';
import { View, Text, StyleSheet, ImageBackground, ActivityIndicator } from 'react-native';
import backgroundImage from '../assets/background.png';

export default function LoadingScreen() {
  return (
    <View style={styles.container}>
      <ImageBackground source={backgroundImage} style={styles.image}>
        <Text style={styles.text}>CraveAI</Text>
        <ActivityIndicator size="large" color="maroon" style={styles.spinner} />
      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  image: {
    flex: 1,
    resizeMode: 'cover',
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'maroon',
    marginBottom: 20, // Add some space between the text and the spinner
  },
  spinner: {
    marginTop: 20, // Add some space between the text and the spinner
  },
});
