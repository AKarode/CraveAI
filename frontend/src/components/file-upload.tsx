"use client"

import { useState, useRef } from "react"
import { Upload, File } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface FileUploadProps {
  onUpload: (file: File) => void
}

export function FileUpload({ onUpload }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    // Check if file is an image or PDF
    if (file.type.startsWith("image/") || file.type === "application/pdf") {
      setSelectedFile(file)
      onUpload(file)
    } else {
      alert("Please upload an image or PDF file")
    }
  }

  const onButtonClick = () => {
    if (inputRef.current) {
      inputRef.current.click()
    }
  }

  return (
    <div className="flex flex-col items-center">
      <div 
        className={`w-full p-6 border-2 border-dashed rounded-lg text-center ${
          dragActive ? "border-primary bg-primary/5" : "border-muted-foreground/20"
        } transition-colors duration-200 cursor-pointer`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
      >
        <Input
          ref={inputRef}
          type="file"
          accept="image/*,application/pdf"
          onChange={handleChange}
          className="hidden"
        />
        
        <div className="flex flex-col items-center gap-2">
          <div className="p-3 rounded-full bg-primary/10">
            <Upload className="h-6 w-6 text-primary" />
          </div>
          <p className="text-sm font-medium">
            Drag and drop or click to upload
          </p>
          <p className="text-xs text-muted-foreground">
            Supports images and PDF files
          </p>
        </div>
      </div>

      {selectedFile && (
        <div className="mt-4 w-full">
          <div className="flex items-center gap-2 p-2 rounded-lg bg-muted">
            <File className="h-5 w-5 text-muted-foreground" />
            <span className="text-sm truncate flex-1">
              {selectedFile.name}
            </span>
            <span className="text-xs text-muted-foreground">
              {(selectedFile.size / 1024).toFixed(0)} KB
            </span>
          </div>
        </div>
      )}
    </div>
  )
} 