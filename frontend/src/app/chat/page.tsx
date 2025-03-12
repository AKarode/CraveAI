"use client"

import { useState, useRef, useEffect } from "react"
import Link from "next/link"
import { ArrowLeft, Send, Camera, User, Bot } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { toast } from "sonner"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Check if there's menu text from the scan page
    const menuText = localStorage.getItem("menuText")
    if (menuText) {
      // Add a system message with the menu text
      setMessages([
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "I've analyzed the menu you scanned. What would you like to know about these items? If you have any dietary preferences or restrictions, I can help you find suitable options.",
          timestamp: new Date()
        }
      ])
      
      // Clear the localStorage
      localStorage.removeItem("menuText")
    } else {
      // Add a welcome message if there's no menu text
      setMessages([
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "Hello! I'm your food assistant. I can help you with menu recommendations based on your dietary preferences. You can scan a menu or ask me questions about food items.",
          timestamp: new Date()
        }
      ])
    }
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  const handleSendMessage = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      // This is a placeholder for the actual API call
      // In a real implementation, you would send the message to your backend
      // and get the response from the LLM
      setTimeout(() => {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: generateResponse(input),
          timestamp: new Date()
        }
        setMessages(prev => [...prev, assistantMessage])
        setIsLoading(false)
      }, 1500)
    } catch (error) {
      console.error("Error sending message:", error)
      setIsLoading(false)
      toast.error("Failed to send message. Please try again.")
    }
  }

  const generateResponse = (query: string) => {
    // This is a placeholder for the actual LLM response
    // In a real implementation, this would come from your backend
    if (query.toLowerCase().includes("vegetarian")) {
      return "Based on the menu, I recommend the Caesar Salad without the chicken. The Margherita Pizza is also vegetarian-friendly."
    } else if (query.toLowerCase().includes("gluten")) {
      return "For gluten-free options, I recommend the Grilled Salmon. The Caesar Salad without croutons would also be suitable."
    } else if (query.toLowerCase().includes("recommend") || query.toLowerCase().includes("suggestion")) {
      return "I'd recommend the Spaghetti Carbonara - it's a popular dish. If you're looking for something lighter, the Caesar Salad is a good option."
    } else {
      return "I'm here to help with menu recommendations. You can ask me about specific dietary needs like vegetarian or gluten-free options, or ask for general recommendations."
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <main className="flex flex-col h-screen">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center">
          <Link href="/">
            <Button variant="ghost" size="icon" className="mr-2">
              <ArrowLeft className="h-5 w-5" />
            </Button>
          </Link>
          <h1 className="text-xl font-bold">Chat with AI</h1>
          <div className="ml-auto">
            <Link href="/scan">
              <Button variant="outline" size="sm">
                <Camera className="h-4 w-4 mr-2" />
                Scan Menu
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="container max-w-2xl mx-auto space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div className="flex items-start gap-3 max-w-[80%]">
                {message.role === "assistant" && (
                  <Avatar>
                    <AvatarFallback>AI</AvatarFallback>
                    <AvatarImage src="/bot-avatar.png" />
                  </Avatar>
                )}
                <Card className={message.role === "user" ? "bg-primary text-primary-foreground" : ""}>
                  <CardContent className="p-3">
                    <p className="whitespace-pre-wrap">{message.content}</p>
                    <div className="mt-1 text-xs opacity-70 text-right">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </CardContent>
                </Card>
                {message.role === "user" && (
                  <Avatar>
                    <AvatarFallback>
                      <User className="h-4 w-4" />
                    </AvatarFallback>
                    <AvatarImage src="/user-avatar.png" />
                  </Avatar>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="flex items-start gap-3 max-w-[80%]">
                <Avatar>
                  <AvatarFallback>AI</AvatarFallback>
                  <AvatarImage src="/bot-avatar.png" />
                </Avatar>
                <Card>
                  <CardContent className="p-3">
                    <div className="flex space-x-2">
                      <div className="h-2 w-2 bg-muted-foreground/30 rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></div>
                      <div className="h-2 w-2 bg-muted-foreground/30 rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></div>
                      <div className="h-2 w-2 bg-muted-foreground/30 rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="border-t p-4">
        <div className="container max-w-2xl mx-auto">
          <div className="flex items-center gap-2">
            <Input
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              className="flex-1"
            />
            <Button 
              onClick={handleSendMessage} 
              disabled={!input.trim() || isLoading}
              size="icon"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </main>
  )
} 