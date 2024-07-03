import React from "react";
import {
  View,
  Text,
  StyleSheet,
  ImageBackground,
  Image,
  TouchableOpacity,
  Dimensions,
} from "react-native";
import { useNavigation } from "@react-navigation/native";
import backgroundImage from "../assets/background.png";
import middleImage from "../assets/food.jpg"; // Replace this with your actual image

const { width, height } = Dimensions.get("window");

export default function HomeScreen() {
  const navigation = useNavigation();

  const navigateToChat = () => {
    navigation.navigate("ChatScreen");
  };

  const navigateToScan = () => {
    navigation.navigate("ScanScreen");
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
          <TouchableOpacity style={styles.button} onPress={navigateToScan}>
            <Text style={styles.buttonText}>SCAN MENU</Text>
          </TouchableOpacity>
        </View>
      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000000",
  },
  image: {
    flex: 1,
    resizeMode: "cover",
    width: "100%",
    height: "100%",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000000",
  },
  title: {
    fontSize: width * 0.12, // Relative font size
    fontWeight: "bold",
    color: "white",
    marginTop: height * 0.1, // Relative margin
    textAlign: "center",
  },
  middleContainer: {
    flex: 1,
    marginTop: height * 0.05,
    alignItems: "center", // Center items horizontally
  },
  middleImage: {
    width: width * 0.90, // Relative width
    height: height * 0.26, // Relative height
    borderRadius: 15,
  },
  buttonContainer: {
    flexDirection: "column",
    marginBottom: height * 0.08, // Adjust as needed for spacing
    alignItems: "center", // Center button horizontally
  },
  button: {
    backgroundColor: "#FFFFFF",
    paddingVertical: height * 0.015, // Relative padding
    paddingHorizontal: width * 0.1, // Relative padding
    borderRadius: 15,
    marginVertical: width * 0.05, // Add horizontal margin to space out buttons
  },
  buttonText: {
    fontSize: width * 0.06,
    color: "black",
    fontWeight: "bold",
  },
});
