"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import Webcam from "react-webcam"
import { Camera } from "lucide-react"
import { Button } from "@/components/ui/button"

interface CameraCaptureProps {
  onCapture: (imageData: string) => void
}

export function CameraCapture({ onCapture }: CameraCaptureProps) {
  const webcamRef = useRef<Webcam>(null)
  const [isCameraReady, setIsCameraReady] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    // Check if the device is mobile
    const checkMobile = () => {
      setIsMobile(/iPhone|iPad|iPod|Android/i.test(navigator.userAgent))
    }
    
    checkMobile()
    window.addEventListener("resize", checkMobile)
    
    return () => {
      window.removeEventListener("resize", checkMobile)
    }
  }, [])

  const handleUserMedia = useCallback(() => {
    setIsCameraReady(true)
  }, [])

  const captureImage = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot()
      if (imageSrc) {
        onCapture(imageSrc)
      }
    }
  }, [onCapture])

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: isMobile ? "environment" : "user"
  }

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-full aspect-video bg-muted rounded-lg overflow-hidden mb-4">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          onUserMedia={handleUserMedia}
          className="w-full h-full object-cover"
        />
        {!isCameraReady && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/80">
            <p className="text-sm text-muted-foreground">Loading camera...</p>
          </div>
        )}
      </div>
      <Button 
        onClick={captureImage} 
        disabled={!isCameraReady}
        className="w-full"
      >
        <Camera className="mr-2 h-4 w-4" /> Capture Photo
      </Button>
    </div>
  )
} 