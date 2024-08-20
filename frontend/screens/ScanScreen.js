import React, { useState } from 'react';
import { ImageBackground, View, Text, StyleSheet, TouchableOpacity, TextInput, Button, FlatList, Alert, Modal, Linking } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import backgroundImage from '../assets/background.png';

const ChatScreen = () => {
  const navigation = useNavigation();
  const [searchTerm, setSearchTerm] = useState('');
  const [location, setLocation] = useState('San Francisco'); // Default location or get from user input
  const [restaurants, setRestaurants] = useState([]);
  const [selectedRestaurant, setSelectedRestaurant] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);

  const handleBackPress = () => {
    navigation.goBack();
  };

  const handleSearch = () => {
    const apiKey = 'Op-Nw6i9-9q4sE1uZ-rKU-wHtd7SIwZiQWxl-8-lcLNjr9dp-VobY81Uv4VlK6OZUHjaFwvu7KyqkeLy1ElRCoNazM7TtlNSv5tsEzGtEzRINVdi85VlsBMgfJW6ZnYx'; // Replace with your actual API key

    fetch(`https://api.yelp.com/v3/businesses/search?term=${searchTerm}&location=${location}&limit=10`, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    })
      .then(response => response.json())
      .then(data => {
        console.log('Yelp API response:', data);
        if (data.businesses && Array.isArray(data.businesses)) {
          setRestaurants(data.businesses);
        } else {
          Alert.alert('No Results', 'No restaurants found for your search.');
        }
      })
      .catch(error => {
        Alert.alert('Error', 'Failed to fetch restaurant data');
        console.error('Error fetching data:', error);
      });
  };

  const handleInputChange = (text) => {
    setSearchTerm(text);
  };

  const handleRestaurantPress = (restaurant) => {
    setSelectedRestaurant(restaurant);
    setModalVisible(true);
  };

  const closeModal = () => {
    setSelectedRestaurant(null);
    setModalVisible(false);
  };

  const formatUrl = (name, location) => {
    // Remove punctuation and convert to lowercase
    const formattedName = name.toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, '-');
    const formattedLocation = location.toLowerCase().replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, '-');
    return `https://www.yelp.com/menu/${formattedName}-${formattedLocation}`;
  };

  const openMenuLink = (url) => {
    Linking.openURL(url).catch(err => console.error("Failed to open URL", err));
  };

  const sendRestaurantData = (restaurant) => {
    // Generate the correct menu URL using the formatUrl function
    const menuUrl = formatUrl(restaurant.name, restaurant.location.city);
    
    // Prepare the data to be sent, including the formatted menu URL
    const dataToSend = {
      name: restaurant.name,
      rating: restaurant.rating,
      address: `${restaurant.location.address1}, ${restaurant.location.city}`,
      phone: restaurant.phone,
      menuUrl, // This should be the URL you want to send
    };
  
    // Send the data to the server
    fetch('http://localhost:5001/endpoint', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(dataToSend), // Send the data as JSON
    })
      .then(response => response.json())
      .then(data => {
        Alert.alert('Success', 'Restaurant data sent successfully');
      })
      .catch(error => {
        Alert.alert('Error', 'Failed to send restaurant data');
        console.error('Error sending data:', error);
      });
  };
  

  const renderRestaurant = ({ item }) => (
    <TouchableOpacity style={styles.restaurantContainer} onPress={() => handleRestaurantPress(item)}>
      <Text style={styles.restaurantName}>{item.name}</Text>
      <Text style={styles.restaurantInfo}>Rating: {item.rating}</Text>
      <Text style={styles.restaurantInfo}>Address: {item.location.address1}, {item.location.city}</Text>
      <Text style={styles.restaurantInfo}>Phone: {item.phone}</Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <ImageBackground source={backgroundImage} style={styles.image}>
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="Enter restaurant name"
            value={searchTerm}
            onChangeText={handleInputChange}
          />
          <TextInput
            style={styles.input}
            placeholder="Enter location"
            value={location}
            onChangeText={setLocation}
          />
        </View>
        <Button title="Search" onPress={handleSearch} />
        <FlatList
          data={restaurants}
          renderItem={renderRestaurant}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.list}
        />
      </ImageBackground>

      {selectedRestaurant && (
        <Modal
          visible={modalVisible}
          transparent={true}
          animationType="slide"
          onRequestClose={closeModal}
        >
          <View style={styles.modalContainer}>
            <View style={styles.modalContent}>
              <Text style={styles.modalTitle}>{selectedRestaurant.name}</Text>
              <Text style={styles.modalInfo}>Rating: {selectedRestaurant.rating}</Text>
              <Text style={styles.modalInfo}>Address: {selectedRestaurant.location.address1}, {selectedRestaurant.location.city}</Text>
              <Text style={styles.modalInfo}>Phone: {selectedRestaurant.phone}</Text>
              <Text style={styles.modalInfo}>Menu Link:</Text>
              <TouchableOpacity onPress={() => openMenuLink(formatUrl(selectedRestaurant.name, selectedRestaurant.location.city))}>
                <Text style={styles.menuLink}>
                  {formatUrl(selectedRestaurant.name, selectedRestaurant.location.city)}
                </Text>
              </TouchableOpacity>
              <Button title="Confirm Restaurant" onPress={() => sendRestaurantData(selectedRestaurant)} />
              <Button title="Close" onPress={closeModal} />
            </View>
          </View>
        </Modal>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000000',
  },
  image: {
    flex: 1,
    resizeMode: 'cover',
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000000',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    width: '85%',
    marginTop: 70,
  },
  input: {
    flex: 1,
    backgroundColor: 'white',
    padding: 10,
    borderRadius: 8,
    marginRight: 10,
  },
  list: {
    width: '85%',
    marginTop: 20,
  },
  restaurantContainer: {
    backgroundColor: '#ffffff',
    padding: 10,
    marginVertical: 5,
    borderRadius: 8,
    elevation: 2,
  },
  restaurantName: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  restaurantInfo: {
    fontSize: 14,
    color: '#555',
  },
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalContent: {
    width: '80%',
    padding: 20,
    backgroundColor: '#ffffff',
    borderRadius: 8,
    alignItems: 'center',
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  modalInfo: {
    fontSize: 16,
    color: '#555',
    marginBottom: 5,
  },
  menuLink: {
    fontSize: 16,
    color: 'blue',
    textDecorationLine: 'underline',
    marginVertical: 10,
  },
});

export default ChatScreen;
