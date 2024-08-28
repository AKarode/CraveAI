import React, { useState, useEffect } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { onAuthStateChanged } from "firebase/auth";
import { auth } from './firebaseConfig'; // Import the Firebase auth

// Import screens and navigators
import LoadingScreen from "./screens/LoadingScreen";
import SurveyScreen from "./screens/SurveyScreen";
import WelcomePage from "./screens/WelcomePage";
import LoginScreen from "./screens/LoginScreen";
import ResetPasswordScreen from './screens/ResetPasswordScreen';

import BottomTabsNavigator from './screens/BottomTabsNavigator'; // Import bottom tab navigator

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
        try {
          const userData = await AsyncStorage.getItem(`user-${user.uid}`);
          if (userData === null) {
            await AsyncStorage.setItem(`user-${user.uid}`, "exists");
            setIsNewUser(true);
          } else {
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
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName={user ? (isNewUser ? "WelcomePage" : "BottomTabsNavigator") : "LoginScreen"}
        screenOptions={{
          headerShown: false,
          animation: "none",
        }}
      >
        <Stack.Screen name="LoginScreen" component={LoginScreen} />
        <Stack.Screen name="WelcomePage" component={WelcomePage} />
        <Stack.Screen name="SurveyScreen" component={SurveyScreen} />
        <Stack.Screen name="BottomTabsNavigator" component={BottomTabsNavigator} />
        <Stack.Screen name="ResetPasswordScreen" component={ResetPasswordScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
