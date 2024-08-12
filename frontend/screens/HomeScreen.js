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
import backgroundImage from "../assets/background.png";

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

  return (
    <LinearGradient
      colors={['#ff7e5f', '#feb47b']}
      style={styles.container}
    >
      <ImageBackground source={backgroundImage} style={styles.image}>
        <Text style={styles.title}>CraveAI</Text>
        <View style={styles.middleContainer}>
          {loading ? (
            <ActivityIndicator size="large" color="#FFFFFF" />
          ) : (
            <Image source={{ uri: middleImage }} style={styles.middleImage} />
          )}
        </View>
        {/* The button container has been removed */}
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
    color: "maroon",
    marginTop: height * 0.1,
    textAlign: "center",
  },
  middleContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center", // Center the image container vertically
    paddingVertical: height * 0.05, // Adjust top and bottom spacing here
  },
  middleImage: {
    width: width * 0.90,
    height: height * 0.40,
    borderRadius: 15,
    marginVertical: 45, // Adjust vertical positioning here
  },
});
