import React, { useState } from 'react';
import { View, Text, StyleSheet, Button, ImageBackground, ScrollView, TouchableOpacity } from 'react-native';
import { RadioButton } from 'react-native-paper';

// Import your background image from assets
import backgroundImage from '../assets/background.png';

export default function SurveyScreen({ navigation }) {
  const [answers, setAnswers] = useState({
    question1: [],
    question2: [],
    question3: [],
    question4: [],
    question5: [],
    question6: [],
    question7: [],
  });

  const handleNavigate = () => {
    navigation.replace('HomeScreen');
  };

  const handleAnswerChange = (question, answer) => {
    // Toggle answer selection for the question
    if (answers[question].includes(answer)) {
      // Remove answer if already selected
      setAnswers(prevAnswers => ({
        ...prevAnswers,
        [question]: prevAnswers[question].filter(item => item !== answer),
      }));
    } else {
      // Add answer if not already selected
      setAnswers(prevAnswers => ({
        ...prevAnswers,
        [question]: [...prevAnswers[question], answer],
      }));
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
              answers[question].includes(option.value) && styles.radioButtonSelected,
            ]}
            onPress={() => handleAnswerChange(question, option.value)}
          >
            <RadioButton.Android
              value={option.value}
              status={answers[question].includes(option.value) ? 'checked' : 'unchecked'}
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
    { question: 'question1', questionText: 'Spicyness 🌶️', options: [
      { value: 'option1', label: 'Low' },
      { value: 'option2', label: 'Mild' },
      { value: 'option3', label: 'Hot' },
    ]},
    { question: 'question2', questionText: 'Budget 💸', options: [
      { value: 'option1', label: '$' },
      { value: 'option2', label: '$$' },
      { value: 'option3', label: '$$$' },
    ]},
    { question: 'question3', questionText: 'Favorite Cuisine 😋', options: [
      { value: 'option1', label: 'Asian' },
      { value: 'option2', label: 'Mexican' },
      { value: 'option3', label: 'Italian' },
    ]},
    { question: 'question4', questionText: 'Dietary Preferences 📋', options: [
      { value: 'option1', label: 'None' },
      { value: 'option2', label: 'Gluten-Free' },
      { value: 'option3', label: 'Vegetarian' },
    ]},
    { question: 'question5', questionText: 'Frequency of Exploring New Places 🌎', options: [
      { value: 'option1', label: 'Rarely' },
      { value: 'option2', label: 'Sometimes' },
      { value: 'option3', label: 'Often' },
    ]},
    { question: 'question6', questionText: 'Preferred Cooking Style 👨‍🍳', options: [
      { value: 'option1', label: 'Grilled' },
      { value: 'option2', label: 'Fried' },
      { value: 'option3', label: 'Steamed' },
    ]},
    { question: 'question7', questionText: 'Favorite Dessert 🎂', options: [
      { value: 'option1', label: 'Chocolate' },
      { value: 'option2', label: 'Fruit Based' },
      { value: 'option3', label: 'Pastries' },
    ]},
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
          onPress={handleNavigate}
          color="#ffffff"
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
    color: '#ffffff',
    marginTop: 40,
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
    color: '#ffffff',
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
    backgroundColor: "#000000",
    paddingVertical: 10,
    paddingHorizontal: 30,
    borderRadius: 10,
    width: '80%', // Adjust width to make all buttons the same size
    justifyContent: 'space-between',
  },
  radioButtonSelected: {
    backgroundColor: '#9966CB',
  },
  radioText: {
    color: '#ffffff',
    marginLeft: 8,
    flex: 1,
    textAlign: 'center',
  },
});
