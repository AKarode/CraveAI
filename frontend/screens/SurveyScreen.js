import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Button,
  ImageBackground,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { RadioButton } from 'react-native-paper';
import { doc, setDoc, collection, addDoc} from 'firebase/firestore';
import { auth, db } from '../firebaseConfig'; // Import Firestore and auth
import backgroundImage from '../assets/background.png';

export default function SurveyScreen({ navigation }) {
  const [answers, setAnswers] = useState({
    question1: null,
    question2: null,
    question3: null,
    question4: null,
    question5: null,
    question6: null,
    question7: null,
  });

  const handleNavigate = () => {
    navigation.replace('BottomTabsNavigator'); // Navigate to the BottomTabsNavigator
  };

  const handleAnswerChange = (question, answer) => {
    setAnswers(prevAnswers => ({
      ...prevAnswers,
      [question]: answer
    }));
  };

  const handleSubmit = async () => {
    try {
      const userId = auth.currentUser.uid;
      const surveyDocId = `surveydata_${userId}`;
      const surveyDocRef = doc(db, 'users', userId, 'surveys', surveyDocId);
  
      // Save the survey data with the specified document ID
      await setDoc(surveyDocRef, {
        responses: {
          ...answers,
          timestamp: new Date()
        }
      });
  
      // Log the survey data to the console
      console.log("Survey data saved:", {
        userId,
        responses: {
          ...answers,
          timestamp: new Date()
        }
      });
  
      // Navigate to the home screen or show a success message
      handleNavigate();
    } catch (error) {
      console.error("Error saving survey data:", error);
    }
  };
  
  

  const renderQuestion = (question, questionText, options) => (
    <View style={styles.questionContainer} key={question}>
      <Text style={styles.questionText}>{questionText}</Text>
      <View style={styles.radioContainer}>
        {options.map((option) => (
          <TouchableOpacity
            key={option.value}
            style={[
              styles.radioButton,
              answers[question] === option.value && styles.radioButtonSelected,
            ]}
            onPress={() => handleAnswerChange(question, option.value)}
          >
            <RadioButton.Android
              value={option.value}
              status={answers[question] === option.value ? 'checked' : 'unchecked'}
              color={'#000000'} // Change the color to match your theme
            />
            <Text style={styles.radioText}>{option.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );

  // Define your questions and options
  const questions = [
    {
      question: 'question1',
      questionText: 'Spicyness 🌶️',
      options: [
        { value: 'Low', label: 'Low' },
        { value: 'Mild', label: 'Mild' },
        { value: 'Hot', label: 'Hot' },
      ],
    },
    {
      question: 'question2',
      questionText: 'Budget 💸',
      options: [
        { value: '$', label: '$' },
        { value: '$$', label: '$$' },
        { value: '$$$', label: '$$$' },
      ],
    },
    {
      question: 'question3',
      questionText: 'Favorite Cuisine 😋',
      options: [
        { value: 'Asian', label: 'Asian' },
        { value: 'Mexican', label: 'Mexican' },
        { value: 'Italian', label: 'Italian' },
      ],
    },
    {
      question: 'question4',
      questionText: 'Dietary Preferences 📋',
      options: [
        { value: 'None', label: 'None' },
        { value: 'Gluten-Free', label: 'Gluten-Free' },
        { value: 'Vegetarian', label: 'Vegetarian' },
      ],
    },
    {
      question: 'question5',
      questionText: 'Frequency of Exploring New Places 🌎',
      options: [
        { value: 'Rarely', label: 'Rarely' },
        { value: 'Sometimes', label: 'Sometimes' },
        { value: 'Often', label: 'Often' },
      ],
    },
    {
      question: 'question6',
      questionText: 'Preferred Cooking Style 👨‍🍳',
      options: [
        { value: 'Grilled', label: 'Grilled' },
        { value: 'Fried', label: 'Fried' },
        { value: 'Steamed', label: 'Steamed' },
      ],
    },
    {
      question: 'question7',
      questionText: 'Favorite Dessert 🎂',
      options: [
        { value: 'Chocolate', label: 'Chocolate' },
        { value: 'Fruit Based', label: 'Fruit Based' },
        { value: 'Pastries', label: 'Pastries' },
      ],
    },
  ];
  

  return (
    <ImageBackground source={backgroundImage} style={styles.background}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Getting Started</Text>
        {questions.map(({ question, questionText, options }) =>
          renderQuestion(question, questionText, options)
        )}
        <Button
          title="Complete Survey"
          onPress={handleSubmit}
          color="maroon"
        />
      </ScrollView>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
    resizeMode: 'cover',
    backgroundColor: '#000000',
  },
  container: {
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    paddingVertical: 20,
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: 'maroon',
    marginTop: 50,
    marginBottom: 50,
  },
  questionContainer: {
    marginBottom: 10,
    width: '80%',
    alignItems: 'center',
  },
  questionText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'maroon',
    marginTop: 20,
    textAlign: 'center',
    marginBottom: 25,
  },
  radioContainer: {
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
  },
  radioButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
    backgroundColor: "white",
    paddingVertical: 10,
    paddingHorizontal: 30,
    borderRadius: 10,
    width: '80%', // Adjust width to make all buttons the same size
    justifyContent: 'space-between',
    borderColor: 'maroon', // Maroon color for the border
    borderWidth: 2, // Adjust the border width as needed
    borderRadius: 20 // Make the input fill the height of the container
  },
  radioButtonSelected: {
    backgroundColor: 'maroon',
  },
  radioText: {
    color: 'black',
    marginLeft: 8,
    flex: 1,
    textAlign: 'center',
    fontWeight: 'bold'
  },
});
