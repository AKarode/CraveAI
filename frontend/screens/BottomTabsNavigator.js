import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import HomeScreen from './HomeScreen';
import ScanScreen from './ScanScreen';
import AccountScreen from './AccountScreen';
import ChatScreen from './ChatScreen';

const Tab = createBottomTabNavigator();

export default function BottomTabsNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ color, size }) => {
          let iconName;
          if (route.name === 'Home') {
            iconName = 'home';
          } else if (route.name === 'Chat') {
            iconName = 'chatbubbles';
          } else if (route.name === 'Scan') {
            iconName = 'scan';
          } else if (route.name === 'Account') {
            iconName = 'person';
          }
          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: 'maroon', // Active tab color
        tabBarInactiveTintColor: 'black', // Inactive tab color
        tabBarStyle: {
          height: 70, // Adjust the height here
          paddingBottom: 10, // Add padding to push icons up
        },
        tabBarLabelStyle: {
          fontSize: 12, // Adjust font size if needed
          paddingBottom: 5, // Add padding to push labels up if needed
        },
        headerShown: false, // Hide the header here
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Scan" component={ScanScreen} />
      <Tab.Screen name="Chat" component={ChatScreen} />
      <Tab.Screen name="Account" component={AccountScreen} />
    </Tab.Navigator>
  );
}
