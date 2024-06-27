import React from 'react';
import { View, Text, StyleSheet, ImageBackground } from 'react-native';
import backgroundImage from '../assets/background.png';

export default function HomeScreen() {
  return (
    <View style={styles.container}>
      <ImageBackground source={backgroundImage} style={styles.image}>
        <Text style={styles.text}>Home Screen</Text>
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
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
});
