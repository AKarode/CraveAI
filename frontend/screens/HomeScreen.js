// Import necessary components and libraries
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ImageBackground,
  Image,
  TouchableOpacity,
  Dimensions,
  ActivityIndicator,
} from "react-native";
import { useNavigation } from "@react-navigation/native";
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import backgroundImage from "../assets/background.png";
import { signOut } from "firebase/auth";
import { auth } from "../firebaseConfig";

const { width, height } = Dimensions.get("window");

const UNSPLASH_ACCESS_KEY = "W0miEWOlMyBWF-aeaU4QmSRPLL8lj2Ist_ONNvI97eo"; // Replace with your Unsplash Access Key

export default function HomeScreen() {
  const navigation = useNavigation();
  const [middleImage, setMiddleImage] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchImage = async () => {
      try {
        const response = await axios.get("https://api.unsplash.com/photos/random", {
          headers: {
            Authorization: `Client-ID ${UNSPLASH_ACCESS_KEY}`,
          },
          params: {
            query: "restaurant dish",
            orientation: "landscape",
          },
        });
        const imageUrl = response.data.urls.regular;
        await AsyncStorage.setItem("middleImage", imageUrl);
        await AsyncStorage.setItem("lastFetchedDate", new Date().toISOString().split("T")[0]);
        setMiddleImage(imageUrl);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching image:", error);
        setLoading(false);
      }
    };

    const checkAndFetchImage = async () => {
      const lastFetchedDate = await AsyncStorage.getItem("lastFetchedDate");
      const middleImage = await AsyncStorage.getItem("middleImage");
      const currentDate = new Date().toISOString().split("T")[0];

      if (lastFetchedDate === currentDate && middleImage) {
        setMiddleImage(middleImage);
        setLoading(false);
      } else {
        fetchImage();
      }
    };

    checkAndFetchImage();
  }, []);

  const navigateToChat = () => {
    navigation.navigate("ChatScreen");
  };

  const navigateToScan = () => {
    navigation.navigate("ScanScreen");
  };

  const navigateToAccount = () => {
    navigation.navigate("AccountScreen");
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigation.navigate("LoginScreen");
    } catch (error) {
      console.error("Error signing out:", error);
    }
  };

  return (
    <LinearGradient
      colors={['#ff7e5f', '#feb47b']}
      style={styles.container}
    >
      <ImageBackground source={backgroundImage} style={styles.image}>
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Ionicons name="log-out" size={24} color="white" />
        </TouchableOpacity>
        <Text style={styles.title}>CraveAI</Text>
        <View style={styles.middleContainer}>
          {loading ? (
            <ActivityIndicator size="large" color="#FFFFFF" />
          ) : (
            <Image source={{ uri: middleImage }} style={styles.middleImage} />
          )}
        </View>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={navigateToChat}>
            <Ionicons name="chatbubbles" size={24} color="white" />
            <Text style={styles.buttonText}>Chatbot</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={navigateToScan}>
            <Ionicons name="scan" size={24} color="white" />
            <Text style={styles.buttonText}>Scan Menu</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={navigateToAccount}>
            <Ionicons name="person" size={24} color="white" />
            <Text style={styles.buttonText}>Account Info</Text>
          </TouchableOpacity>
        </View>
      </ImageBackground>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  image: {
    flex: 1,
    resizeMode: "cover",
    width: "100%",
    height: "100%",
    justifyContent: "center",
    alignItems: "center",
  },
  title: {
    fontSize: 48,
    fontWeight: "bold",
    color: "white",
    marginTop: height * 0.1,
    textAlign: "center",
  },
  middleContainer: {
    flex: 1,
    marginTop: height * 0.05,
    alignItems: "center",
  },
  middleImage: {
    width: width * 0.90,
    height: height * 0.40,
    borderRadius: 15,
  },
  buttonContainer: {
    flexDirection: "row",
    justifyContent: "space-around",
    width: "100%",
    marginBottom: height * 0.08,
  },
  button: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#ff7e5f",
    paddingVertical: 15,
    paddingHorizontal: 5,
    borderRadius: 40,
    marginHorizontal: 10,
  },
  buttonText: {
    fontSize: 15,
    color: "white",
    fontWeight: "bold",
    marginLeft: 10,
  },
  logoutButton: {
    position: "absolute",
    marginTop: 10,
    top: 30,
    right: 20,
    backgroundColor: "#ff7e5f",
    padding: 10,
    borderRadius: 30,
  },
});
