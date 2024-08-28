import React, { useState } from 'react';
import { View, Text, StyleSheet, Button, FlatList, Dimensions, ImageBackground } from 'react-native';
import { useNavigation } from '@react-navigation/native';


const WelcomePage = () => {
  const navigation = useNavigation();
  const [activeIndex, setActiveIndex] = useState(0);

  const handleNavigate = () => {
    navigation.navigate('SurveyScreen');
  };

  const renderItem = ({ item }) => (
    <View style={styles.carouselItem}>
      <Text style={styles.carouselText}>{item.text}</Text>
    </View>
  );

  const infoItems = [
    { key: '1', text: 'Please complete our survey for personalized content.' },
    { key: '2', text: 'Chat with our AI powered chatbot for personalized recommendations' },
    { key: '3', text: 'Thank you for choosing CraveAI' },
  ];

  const handleScroll = (event) => {
    const index = Math.round(event.nativeEvent.contentOffset.x / Dimensions.get('window').width);
    setActiveIndex(index);
  };

  return (
    <ImageBackground source={require('../assets/background.png')} style={styles.container}>
      <Text style={styles.title}>Welcome to</Text>
      <Text style={[styles.title, styles.subtitle]}>Crave AI</Text>
      <FlatList
        data={infoItems}
        renderItem={renderItem}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onScroll={handleScroll}
        style={styles.carousel}
      />
      <View style={styles.pagination}>
        {infoItems.map((_, index) => (
          <View
            key={index}
            style={[
              styles.dot,
              { backgroundColor: index === activeIndex ? '#000' : '#ccc' },
            ]}
          />
        ))}
      </View>
      <View style={styles.buttonContainer}>
        <Button
          title="Start Survey"
          onPress={handleNavigate}
          color="maroon" // Button text color
        />
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'flex-start',
    alignItems: 'center',
    paddingTop: 80,
    backgroundColor: '#000000',
  },
  title: {
    fontSize: 44,
    fontWeight: 'bold',
    marginTop: 55,
    fontFamily: 'Lexend',
    color: 'black', // Text color
  },
  subtitle: {
    marginTop: 20,
    fontSize: 44,
    fontFamily: 'Lexend',
    color: 'black', // Text color
  },
  image: {
    flex: 1,
    resizeMode: 'cover',
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  carousel: {
    marginVertical: 20,
  },
  carouselItem: {
    justifyContent: 'center',
    alignItems: 'center',
    width: Dimensions.get('window').width,
    padding: 20,
  },
  carouselText: {
    fontSize: 18,
    textAlign: 'center',
    fontFamily: 'Lexend',
    color: 'black', // Text color
  },
  pagination: {
    flexDirection: 'row',
    marginTop: 10, // Adjusted margin to raise the dots a bit
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginHorizontal: 5,
    marginBottom: 50, // Adjusted margin to position the dots right under the info
    backgroundColor: 'black', // Dot color
  },
  buttonContainer: {
    marginBottom: 50, // Adjusted margin to raise the button higher
  },
});

export default WelcomePage;
