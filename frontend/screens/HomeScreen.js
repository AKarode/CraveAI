import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ImageBackground,
  Image,
  ScrollView,
  ActivityIndicator,
  Dimensions,
} from "react-native";
import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Location from 'expo-location';
import { LinearGradient } from 'expo-linear-gradient';
import backgroundImage from "../assets/background.png";

const { width, height } = Dimensions.get("window");

const UNSPLASH_ACCESS_KEY = "W0miEWOlMyBWF-aeaU4QmSRPLL8lj2Ist_ONNvI97eo"; // Replace with your Unsplash Access Key
const GOOGLE_PLACES_API_KEY = "AIzaSyCxxb2za84ZBpzKp-2KXju2udPW2iRNVOk"; // Replace with your Google Places API Key

export default function HomeScreen() {
  const [middleImage, setMiddleImage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [restaurants, setRestaurants] = useState([]);
  const [locationError, setLocationError] = useState(null);

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

    const fetchRestaurants = async () => {
      try {
        let { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted') {
          setLocationError('Permission to access location was denied');
          setLoading(false);
          return;
        }

        let location = await Location.getCurrentPositionAsync({});
        const latitude = location.coords.latitude;
        const longitude = location.coords.longitude;

        const response = await axios.get(
          `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${latitude},${longitude}&radius=5000&type=restaurant&key=${GOOGLE_PLACES_API_KEY}`
        );

        setRestaurants(response.data.results.slice(0, 10));
      } catch (error) {
        console.error("Error fetching restaurants:", error);
      }
    };

    checkAndFetchImage();
    fetchRestaurants();
  }, []);

  const renderRestaurant = (item) => (
    <View key={item.place_id} style={styles.restaurantContainer}>
      {item.photos && item.photos.length > 0 ? (
        <Image
          source={{ uri: `https://maps.googleapis.com/maps/api/place/photo?maxwidth=100&photoreference=${item.photos[0].photo_reference}&key=${GOOGLE_PLACES_API_KEY}` }}
          style={styles.restaurantImage}
        />
      ) : (
        <View style={styles.noImageContainer}>
          <Text style={styles.noImageText}>No Image</Text>
        </View>
      )}
      <View style={styles.restaurantInfo}>
        <Text style={styles.restaurantName}>{item.name}</Text>
        <Text style={styles.restaurantAddress}>{item.vicinity}</Text>
        <Text style={styles.restaurantRating}>Rating: {item.rating || 'N/A'}</Text>
      </View>
    </View>
  );

  return (
    <LinearGradient
      colors={['#ff7e5f', '#feb47b']}
      style={styles.container}
    >
      <ImageBackground source={backgroundImage} style={styles.image}>
        <ScrollView contentContainerStyle={styles.scrollViewContent}
        showsVerticalScrollIndicator={false}>
          <Text style={styles.title}>CraveAI</Text>
          <View style={styles.middleContainer}>
            {loading ? (
              <ActivityIndicator size="large" color="#FFFFFF" />
            ) : (
              <Image source={{ uri: middleImage }} style={styles.middleImage} />
            )}
          </View>
          <View style={styles.restaurantListContainer}>
            <Text style={styles.sectionTitle}>Top Restaurants Nearby</Text>
            {locationError ? (
              <Text style={styles.errorText}>{locationError}</Text>
            ) : (
              restaurants.map(renderRestaurant)
            )}
          </View>
        </ScrollView>
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
  scrollViewContent: {
    alignItems: 'center',
    paddingVertical: 20,
    paddingRight: 0, // Adjust to move scroll indicator to the right
  },
  title: {
    fontSize: 48,
    fontWeight: "bold",
    color: "maroon",
    marginTop: height * 0.1,
    textAlign: "center",
  },
  middleContainer: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: height * 0.05,
  },
  middleImage: {
    width: width * 0.8,
    height: height * 0.3,
    borderRadius: 15,
    marginVertical: 45,
  },
  restaurantListContainer: {
    width: '100%', // Make the container wider
    paddingVertical: 0,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'maroon',
    marginBottom: 10,
  },
  restaurantContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 2,
    borderBottomColor: 'maroon'
  },
  restaurantImage: {
    width: 100,
    height: 100,
    borderRadius: 10,
    marginRight: 10,
  },
  noImageContainer: {
    width: 100,
    height: 100,
    borderRadius: 10,
    backgroundColor: '#ccc',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  noImageText: {
    color: '#666',
  },
  restaurantInfo: {
    flex: 1,
  },
  restaurantName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'black',
  },
  restaurantAddress: {
    fontSize: 14,
    color: 'black',
    marginTop: 5,
  },
  restaurantRating: {
    fontSize: 14,
    color: 'teal',
    marginTop: 5,
  },
  errorText: {
    color: 'red',
    textAlign: 'center',
  },
});
