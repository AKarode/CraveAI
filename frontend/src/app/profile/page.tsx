"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { ArrowLeft, Save, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { Textarea } from "@/components/ui/textarea"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { toast } from "sonner"

// Define dietary preferences type
type DietaryPreferences = {
  name: string
  vegetarian: boolean
  vegan: boolean
  glutenFree: boolean
  dairyFree: boolean
  nutAllergy: boolean
  shellfishAllergy: boolean
  otherAllergies: string
  favoriteFood: string
  dislikedFood: string
}

// Default preferences
const defaultPreferences: DietaryPreferences = {
  name: "",
  vegetarian: false,
  vegan: false,
  glutenFree: false,
  dairyFree: false,
  nutAllergy: false,
  shellfishAllergy: false,
  otherAllergies: "",
  favoriteFood: "",
  dislikedFood: ""
}

export default function ProfilePage() {
  const [preferences, setPreferences] = useState<DietaryPreferences>(defaultPreferences)
  const [isSaving, setIsSaving] = useState(false)

  // Load preferences from localStorage on component mount
  useEffect(() => {
    const savedPreferences = localStorage.getItem("dietaryPreferences")
    if (savedPreferences) {
      try {
        setPreferences(JSON.parse(savedPreferences))
      } catch (error) {
        console.error("Error parsing saved preferences:", error)
      }
    }
  }, [])

  const handleCheckboxChange = (field: keyof DietaryPreferences) => {
    setPreferences(prev => ({
      ...prev,
      [field]: !prev[field as keyof typeof prev]
    }))
  }

  const handleInputChange = (field: keyof DietaryPreferences, value: string) => {
    setPreferences(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const savePreferences = () => {
    setIsSaving(true)
    
    // Simulate API call with timeout
    setTimeout(() => {
      try {
        localStorage.setItem("dietaryPreferences", JSON.stringify(preferences))
        toast.success("Preferences saved successfully!")
      } catch (error) {
        console.error("Error saving preferences:", error)
        toast.error("Failed to save preferences. Please try again.")
      } finally {
        setIsSaving(false)
      }
    }, 1000)
  }

  return (
    <main className="container max-w-2xl mx-auto py-6 px-4">
      <div className="flex items-center mb-6">
        <Link href="/">
          <Button variant="ghost" size="icon" className="mr-2">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <h1 className="text-2xl font-bold">Your Profile</h1>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-4">
              <Avatar className="h-16 w-16">
                <AvatarFallback className="text-xl">
                  <User className="h-8 w-8" />
                </AvatarFallback>
                <AvatarImage src="/user-avatar.png" />
              </Avatar>
              <div>
                <CardTitle>Dietary Preferences</CardTitle>
                <CardDescription>
                  Set your dietary preferences to get personalized menu recommendations
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="name">Your Name</Label>
              <Input 
                id="name" 
                placeholder="Enter your name" 
                value={preferences.name}
                onChange={(e) => handleInputChange("name", e.target.value)}
              />
            </div>

            <div className="space-y-3">
              <Label>Dietary Restrictions</Label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="vegetarian" 
                    checked={preferences.vegetarian}
                    onCheckedChange={() => handleCheckboxChange("vegetarian")}
                  />
                  <Label htmlFor="vegetarian" className="cursor-pointer">Vegetarian</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="vegan" 
                    checked={preferences.vegan}
                    onCheckedChange={() => handleCheckboxChange("vegan")}
                  />
                  <Label htmlFor="vegan" className="cursor-pointer">Vegan</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="glutenFree" 
                    checked={preferences.glutenFree}
                    onCheckedChange={() => handleCheckboxChange("glutenFree")}
                  />
                  <Label htmlFor="glutenFree" className="cursor-pointer">Gluten Free</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="dairyFree" 
                    checked={preferences.dairyFree}
                    onCheckedChange={() => handleCheckboxChange("dairyFree")}
                  />
                  <Label htmlFor="dairyFree" className="cursor-pointer">Dairy Free</Label>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <Label>Allergies</Label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="nutAllergy" 
                    checked={preferences.nutAllergy}
                    onCheckedChange={() => handleCheckboxChange("nutAllergy")}
                  />
                  <Label htmlFor="nutAllergy" className="cursor-pointer">Nut Allergy</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="shellfishAllergy" 
                    checked={preferences.shellfishAllergy}
                    onCheckedChange={() => handleCheckboxChange("shellfishAllergy")}
                  />
                  <Label htmlFor="shellfishAllergy" className="cursor-pointer">Shellfish Allergy</Label>
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="otherAllergies">Other Allergies</Label>
                <Textarea 
                  id="otherAllergies" 
                  placeholder="List any other allergies or sensitivities"
                  value={preferences.otherAllergies}
                  onChange={(e) => handleInputChange("otherAllergies", e.target.value)}
                />
              </div>
            </div>

            <div className="space-y-3">
              <Label>Food Preferences</Label>
              <div className="space-y-2">
                <Label htmlFor="favoriteFood">Favorite Foods</Label>
                <Textarea 
                  id="favoriteFood" 
                  placeholder="What foods do you enjoy?"
                  value={preferences.favoriteFood}
                  onChange={(e) => handleInputChange("favoriteFood", e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dislikedFood">Disliked Foods</Label>
                <Textarea 
                  id="dislikedFood" 
                  placeholder="What foods do you avoid?"
                  value={preferences.dislikedFood}
                  onChange={(e) => handleInputChange("dislikedFood", e.target.value)}
                />
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button 
              onClick={savePreferences} 
              disabled={isSaving}
              className="w-full"
            >
              {isSaving ? "Saving..." : "Save Preferences"}
              {!isSaving && <Save className="ml-2 h-4 w-4" />}
            </Button>
          </CardFooter>
        </Card>
      </div>
    </main>
  )
} 