import React, { useState, useRef, useEffect } from 'react';
import { ImageBackground, View, Text, StyleSheet, TouchableOpacity, TextInput, ScrollView, KeyboardAvoidingView } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import backgroundImage from '../assets/background.png';

export default function ChatScreen() {
  const navigation = useNavigation();
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const scrollViewRef = useRef();

  const handleBackPress = () => {
    navigation.navigate('HomeScreen'); // Navigate to HomeScreen
  };

  const sendMessage = async () => {
    if (!message.trim()) return; // Don't send empty messages

    const userMessage = message; // Save the user's message
    setChatHistory([...chatHistory, { sender: 'user', message: userMessage }]);
    setMessage('');

    try {
      const payload = { text: userMessage, image_url: 'https://example.com/your_image_url_here' };
      console.log('Sending message:', JSON.stringify(payload)); // Log the payload for debugging
  
      const response = await fetch('http://localhost:5000/process_menu', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
  
      const data = await response.json();
      console.log('Received response:', data); // Log the response for debugging
  
      setChatHistory((prevHistory) => [...prevHistory, { sender: 'bot', message: data.reply }]);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  useEffect(() => {
    // Scroll to the bottom when chatHistory changes
    scrollViewRef.current?.scrollToEnd({ animated: true });
  }, [chatHistory]);

  return (
    <View style={styles.container}>
      <ImageBackground source={backgroundImage} style={styles.image}>
        <Text style={styles.text}>CraveAI ChatBot</Text>

        <ScrollView
          style={styles.chatContainer}
          ref={scrollViewRef}
          onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
        >
          {chatHistory.map((chat, index) => (
            <View
              key={index}
              style={[
                styles.messageBubble,
                chat.sender === 'user' ? styles.userBubble : styles.botBubble,
              ]}
            >
              <Text style={styles.messageText}>{chat.message}</Text>
            </View>
          ))}
        </ScrollView>

        <KeyboardAvoidingView behavior="padding" style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="Type a message..."
            value={message}
            onChangeText={setMessage}
            onSubmitEditing={sendMessage}
          />
          <TouchableOpacity style={styles.sendButton} onPress={sendMessage}>
            <Ionicons name="send" size={24} color="white" />
          </TouchableOpacity>
        </KeyboardAvoidingView>
      </ImageBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  image: {
    flex: 1,
    resizeMode: 'cover',
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  backButton: {
    position: 'absolute',
    top: 40,
    left: 20,
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'maroon',
    marginBottom: 20,
    marginTop: 80,
  },
  chatContainer: {
    flex: 1,
    width: '100%',
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  messageBubble: {
    padding: 10,
    borderRadius: 10,
    marginBottom: 10,
    maxWidth: '80%',
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: 'maroon',
  },
  botBubble: {
    alignSelf: 'flex-start',
    backgroundColor: 'grey',
  },
  messageText: {
    color: 'white',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: 20,
    marginBottom: 10, // Adjust this value to move the input box higher
    height: 60, 
  } ,
  
  input: {
    flex: 1,
    backgroundColor: 'white',
    borderRadius: 20,
    paddingHorizontal: 15,
    marginRight: 10,
    height: '100%',
    borderColor: 'maroon', // Maroon color for the border
    borderWidth: 2, // Adjust the border width as needed
    borderRadius: 20 // Make the input fill the height of the container
  },
  sendButton: {
    backgroundColor: 'maroon',
    borderRadius: 20,
    padding: 10,
    height: '100%', // Make the button fill the height of the container
    justifyContent: 'center',
  },
});
