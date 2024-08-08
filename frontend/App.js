import React, { useState, useEffect } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { ImageBackground, StyleSheet } from "react-native";
import { onAuthStateChanged } from "firebase/auth";

import LoadingScreen from "./screens/LoadingScreen";
import SurveyScreen from "./screens/SurveyScreen";
import HomeScreen from "./screens/HomeScreen";
import WelcomePage from "./screens/WelcomePage";
import ChatScreen from "./screens/ChatScreen";
import ScanScreen from "./screens/ScanScreen";
import LoginScreen from "./screens/LoginScreen";
import AccountScreen from './screens/AccountScreen';
import ResetPasswordScreen from './screens/ResetPasswordScreen';
import { auth } from './firebaseConfig'; // Import the Firebase auth

const Stack = createNativeStackNavigator();

export default function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [isFirstLaunch, setIsFirstLaunch] = useState(null);
  const [user, setUser] = useState(null);
  const [isNewUser, setIsNewUser] = useState(false);

  useEffect(() => {
    async function checkFirstLaunch() {
      try {
        const value = await AsyncStorage.getItem("hasLaunched");
        if (value === null) {
          await AsyncStorage.setItem("hasLaunched", "true");
          setIsFirstLaunch(true);
        } else {
          setIsFirstLaunch(false);
        }
      } catch (error) {
        console.error("Error checking app launch status", error);
        setIsFirstLaunch(true);
      } finally {
        setTimeout(() => {
          setIsLoading(false);
        }, 2000);
      }
    }
    checkFirstLaunch();

    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        // Check if user is new
        try {
          const userData = await AsyncStorage.getItem(`user-${user.uid}`);
          if (userData === null) {
            // User is new
            await AsyncStorage.setItem(`user-${user.uid}`, "exists");
            setIsNewUser(true);
          } else {
            // Existing user
            setIsNewUser(false);
          }
        } catch (error) {
          console.error("Error checking user data", error);
          setIsNewUser(false);
        }
        setUser(user);
      } else {
        setUser(null);
        setIsNewUser(false); // Ensure flag is reset if no user is logged in
      }
    });

    return unsubscribe;
  }, []);

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <ImageBackground
      source={require("./assets/background.png")}
      style={styles.backgroundImage}
    >
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName={user ? (isNewUser ? "WelcomePage" : "HomeScreen") : "LoginScreen"}
          screenOptions={{
            headerShown: false,
            animation: "none",
          }}
        >
          <Stack.Screen name="LoginScreen" component={LoginScreen} />
          <Stack.Screen name="WelcomePage" component={WelcomePage} />
          <Stack.Screen name="SurveyScreen" component={SurveyScreen} />
          <Stack.Screen name="HomeScreen" component={HomeScreen} />
          <Stack.Screen name="ChatScreen" component={ChatScreen} />
          <Stack.Screen name="ScanScreen" component={ScanScreen} />
          <Stack.Screen name="AccountScreen" component={AccountScreen} />
          <Stack.Screen name="ResetPasswordScreen" component={ResetPasswordScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  backgroundImage: {
    flex: 1,
    resizeMode: "cover",
    justifyContent: "center",
  },
});
