"use client"

import { useState, useRef } from "react"
import Link from "next/link"
import { Camera, Upload, ArrowLeft, Scan } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CameraCapture } from "@/components/camera-capture"
import { FileUpload } from "@/components/file-upload"

export default function ScanPage() {
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [extractedText, setExtractedText] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleImageCapture = (imageData: string) => {
    setCapturedImage(imageData)
    setExtractedText(null)
  }

  const handleFileUpload = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      if (e.target?.result) {
        setCapturedImage(e.target.result as string)
        setExtractedText(null)
      }
    }
    reader.readAsDataURL(file)
  }

  const processImage = async () => {
    if (!capturedImage) return

    setIsProcessing(true)
    
    try {
      // This is a placeholder for the actual API call
      // In a real implementation, you would send the image to your backend
      // and get the extracted text back
      setTimeout(() => {
        setExtractedText("Sample menu items extracted from the image:\n\n1. Margherita Pizza - $12.99\n2. Caesar Salad - $8.99\n3. Spaghetti Carbonara - $14.99\n4. Grilled Salmon - $18.99\n5. Chocolate Cake - $6.99")
        setIsProcessing(false)
      }, 2000)
    } catch (error) {
      console.error("Error processing image:", error)
      setIsProcessing(false)
    }
  }

  const handleContinue = () => {
    // In a real implementation, you would send the extracted text to your chat page
    // and redirect the user there
    if (extractedText) {
      // Store the extracted text in localStorage or state management
      localStorage.setItem("menuText", extractedText)
      // Redirect to chat page
      window.location.href = "/chat"
    }
  }

  const resetCapture = () => {
    setCapturedImage(null)
    setExtractedText(null)
  }

  return (
    <main className="container max-w-md mx-auto py-6 px-4">
      <div className="flex items-center mb-6">
        <Link href="/">
          <Button variant="ghost" size="icon" className="mr-2">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <h1 className="text-2xl font-bold">Scan Menu</h1>
      </div>

      {!capturedImage ? (
        <Tabs defaultValue="camera" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-4">
            <TabsTrigger value="camera">Camera</TabsTrigger>
            <TabsTrigger value="upload">Upload</TabsTrigger>
          </TabsList>
          <TabsContent value="camera" className="mt-0">
            <Card>
              <CardContent className="p-4">
                <CameraCapture onCapture={handleImageCapture} />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="upload" className="mt-0">
            <Card>
              <CardContent className="p-4">
                <FileUpload onUpload={handleFileUpload} />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      ) : (
        <div className="space-y-4">
          <Card>
            <CardContent className="p-4">
              <div className="aspect-video relative overflow-hidden rounded-lg">
                <img 
                  src={capturedImage} 
                  alt="Captured menu" 
                  className="w-full h-full object-cover"
                />
              </div>
              
              {!extractedText ? (
                <div className="mt-4 flex flex-col gap-2">
                  <Button 
                    onClick={processImage} 
                    disabled={isProcessing}
                    className="w-full"
                  >
                    {isProcessing ? "Processing..." : "Extract Menu Items"}
                    {!isProcessing && <Scan className="ml-2 h-4 w-4" />}
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={resetCapture}
                    className="w-full"
                  >
                    Take Another Photo
                  </Button>
                </div>
              ) : (
                <div className="mt-4 space-y-4">
                  <div className="p-3 bg-muted rounded-lg max-h-60 overflow-y-auto">
                    <pre className="text-sm whitespace-pre-wrap">{extractedText}</pre>
                  </div>
                  <div className="flex flex-col gap-2">
                    <Button onClick={handleContinue} className="w-full">
                      Continue to Chat
                    </Button>
                    <Button 
                      variant="outline" 
                      onClick={resetCapture}
                      className="w-full"
                    >
                      Scan Another Menu
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </main>
  )
} 