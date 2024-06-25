import React from 'react';
import { View, Text, StyleSheet, Button } from 'react-native';

export default function SurveyScreen({ navigation }) {
  const handleNavigate = () => {
    navigation.navigate('HomeScreen');
  };

  return (
    <View style={styles.container}>
      <Text style={styles.text}>Survey Screen</Text>
      <Button
        title="Complete Survey"
        onPress={handleNavigate}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
  },
});
