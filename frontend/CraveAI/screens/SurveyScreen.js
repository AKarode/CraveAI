import React from 'react';
import { View, Text, StyleSheet, Button, ImageBackground } from 'react-native';

// Import your background image from assets
import backgroundImage from '../assets/background.png';

export default function SurveyScreen({ navigation }) {
  const handleNavigate = () => {
    navigation.replace('HomeScreen');
  };

  return (
    <ImageBackground source={backgroundImage} style={styles.background}>
      <View style={styles.container}>
        <Text style={styles.text}>Survey Screen</Text>
        <Button
          title="Complete Survey"
          onPress={handleNavigate}
          color="#ffffff" // Button text color
        />
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
    resizeMode: 'cover',
    justifyContent: 'center',
    alignItems: 'center',
  },
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  // Semi-transparent black background for readability
    width: '100%',
    height: '100%',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff', // Text color
    marginBottom: 20,
  },
});
