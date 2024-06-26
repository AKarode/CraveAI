import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import AsyncStorage from '@react-native-async-storage/async-storage';

import LoadingScreen from './screens/LoadingScreen';
import WelcomePage from './screens/WelcomePage';
import SurveyScreen from './screens/SurveyScreen';
import HomeScreen from './screens/HomeScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [isFirstLaunch, setIsFirstLaunch] = useState(null);

  useEffect(() => {
    async function checkFirstLaunch() {
      try {
        const value = await AsyncStorage.getItem('hasLaunched');
        if (value === null) {
          await AsyncStorage.setItem('hasLaunched', 'true');
          setIsFirstLaunch(true);
        } else {
          setIsFirstLaunch(false);
        }
      } catch (error) {
        console.error('Error checking app launch status', error);
        setIsFirstLaunch(true); // Default to true if error occurs
      } finally {
        // Simulate a 4-second delay
        setTimeout(() => {
          setIsLoading(false);
        }, 4000); // 4000 milliseconds = 4 seconds
      }
    }
    checkFirstLaunch();

    
  }, []);
  

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName={isFirstLaunch ? 'WelcomePage' : 'HomeScreen'}
        screenOptions={{
          headerShown: false,
          animation: 'none', // Disable animation globally
        }}
      >
        <Stack.Screen name="WelcomePage" component={WelcomePage} />
        <Stack.Screen name="Survey" component={SurveyScreen} />
        <Stack.Screen name="HomeScreen" component={HomeScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
