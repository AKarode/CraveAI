import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert
} from 'react-native';

import { useNavigation } from '@react-navigation/native';

import { signOut } from "firebase/auth";
import { auth } from "../firebaseConfig";
export default function AccountScreen() {
  const navigation = useNavigation();
  const [user, setUser] = useState(null);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(setUser);
    return () => unsubscribe();
  }, []);

  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigation.navigate("LoginScreen");
    } catch (error) {
      console.error("Error signing out:", error);
      Alert.alert("Error", "Failed to sign out. Please try again.");
    }
  };

  if (!user) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Loading...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Account Information</Text>
      <Text style={styles.label}>Email:</Text>
      <Text style={styles.info}>{user.email}</Text>
      <Text style={styles.label}>User ID:</Text>
      <Text style={styles.info}>{user.uid}</Text>
      <Text style={styles.label}>Account Created:</Text>
      <Text style={styles.info}>{user.metadata.creationTime}</Text>
      <TouchableOpacity onPress={handleLogout}>
        <Text style={styles.logoutText}>Logout</Text>
      </TouchableOpacity>
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
  logoutText: {
    fontSize: 18,
    color: 'maroon',
    marginTop: 20,
    textAlign: 'center',
    fontWeight: 'bold'
  },
});
