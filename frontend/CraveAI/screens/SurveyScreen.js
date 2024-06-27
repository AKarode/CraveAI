import React, { useState } from 'react';
import { View, Text, StyleSheet, Button, ImageBackground, ScrollView, TouchableOpacity } from 'react-native';
import { RadioButton } from 'react-native-paper';

// Import your background image from assets
import backgroundImage from '../assets/background.png';

export default function SurveyScreen({ navigation }) {
  const [answers, setAnswers] = useState({
    question1: '',
    question2: '',
    question3: '',
    question4: '',
    question5: '',
  });

  const handleNavigate = () => {
    navigation.replace('HomeScreen');
  };

  const handleAnswerChange = (question, answer) => {
    setAnswers({ ...answers, [question]: answer });
  };

  const renderQuestion = (question, questionText) => (
    <View style={styles.questionContainer} key={question}>
      <Text style={styles.questionText}>{questionText}</Text>
      <RadioButton.Group
        onValueChange={(newValue) => handleAnswerChange(question, newValue)}
        value={answers[question]}
      >
        <View style={styles.radioContainer}>
          <TouchableOpacity
            style={[
              styles.radioButton,
              answers[question] === 'option1' && styles.radioButtonSelected,
            ]}
            onPress={() => handleAnswerChange(question, 'option1')}
          >
            <RadioButton value="option1" />
            <Text style={styles.radioText}>Option 1</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.radioButton,
              answers[question] === 'option2' && styles.radioButtonSelected,
            ]}
            onPress={() => handleAnswerChange(question, 'option2')}
          >
            <RadioButton value="option2" />
            <Text style={styles.radioText}>Option 2</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.radioButton,
              answers[question] === 'option3' && styles.radioButtonSelected,
            ]}
            onPress={() => handleAnswerChange(question, 'option3')}
          >
            <RadioButton value="option3" />
            <Text style={styles.radioText}>Option 3</Text>
          </TouchableOpacity>
        </View>
      </RadioButton.Group>
    </View>
  );

  return (
    <ImageBackground source={backgroundImage} style={styles.background}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Survey Screen</Text>
        {renderQuestion('question1', 'Question 1')}
        {renderQuestion('question2', 'Question 2')}
        {renderQuestion('question3', 'Question 3')}
        {renderQuestion('question4', 'Question 4')}
        {renderQuestion('question5', 'Question 5')}
        <Button
          title="Complete Survey"
          onPress={handleNavigate}
          color="#ffffff" // Button text color
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
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 40,
  },
  questionContainer: {
    marginBottom: 10,
    width: '80%',
    alignItems: 'flex-start',
  },
  questionText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 40,
  },
  radioContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
  },
  radioButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
    backgroundColor: '#000000',
    padding: 10,
    borderRadius: 20,
    flex: 1,
    justifyContent: 'center',
    marginHorizontal: 5,
  },
  radioButtonSelected: {
    backgroundColor: '#444444',
  },
  radioText: {
    color: '#ffffff',
    marginLeft: 10,
  },
});

