import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Button,
  TouchableOpacity,
  Alert
} from 'react-native';
import { auth } from '../firebaseConfig';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';

export default function AccountScreen() {
  const navigation = useNavigation();
  const [user, setUser] = useState(null);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(setUser);
    return () => unsubscribe();
  }, []);

  if (!user) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Loading...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('HomeScreen')}>
        <Ionicons name="arrow-back" size={24} color="white" />
      </TouchableOpacity>
      <Text style={styles.title}>Account Information</Text>
      <Text style={styles.label}>Email:</Text>
      <Text style={styles.info}>{user.email}</Text>
      <Text style={styles.label}>User ID:</Text>
      <Text style={styles.info}>{user.uid}</Text>
      <Text style={styles.label}>Account Created:</Text>
      <Text style={styles.info}>{user.metadata.creationTime}</Text>
      <Button
        title="Reset Password"
        onPress={() => navigation.navigate('ResetPasswordScreen')}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  label: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  info: {
    fontSize: 18,
    marginBottom: 15,
  },
  backButton: {
    position: 'absolute',
    top: 40,
    left: 20,
    backgroundColor: '#ff7e5f',
    padding: 10,
    borderRadius: 30,
    zIndex: 1,
  },
});
