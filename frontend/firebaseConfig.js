import { initializeApp } from 'firebase/app';
import { getAuth, initializeAuth, getReactNativePersistence } from 'firebase/auth';
import AsyncStorage from '@react-native-async-storage/async-storage'; // Import AsyncStorage
import { getFirestore } from 'firebase/firestore';

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyASCaAX4uu_02He-XwcHLsgNyOocu86dPk",
  authDomain: "craveai-fe879.firebaseapp.com",
  projectId: "craveai-fe879",
  storageBucket: "craveai-fe879.appspot.com",
  messagingSenderId: "761496773170",
  appId: "1:761496773170:web:2751a828eb7ebb0905122b",
  measurementId: "G-7MVWNYGS9K"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Auth with AsyncStorage for persistence
const auth = initializeAuth(app, {
  persistence: getReactNativePersistence(AsyncStorage)
});

// Initialize Firestore
const db = getFirestore(app);

export { auth, db };
