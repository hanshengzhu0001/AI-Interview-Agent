import React, { useState } from 'react'
import { Input, Button, Box, VStack, Text, IconButton, Flex, HStack } from '@chakra-ui/react'
import axios from 'axios'
import { FaMicrophone } from 'react-icons/fa'

const Chat: React.FC = () => {
  const [question, setQuestion] = useState<string>("")
  const [messages, setMessages] = useState<{ q: string, a: string }[]>([])

  // Handle text input
  const handleTextInput = async () => {
    if (!question) return

    // Add user message to the chat
    const newMessage = { q: question, a: "" }
    setMessages(prevMessages => [...prevMessages, newMessage])

    try {
      // Send user message to the Flask backend
      const response = await axios.post("http://127.0.0.1:5000/rasa", { message: question })
      if (response.data && response.data.response) {
        const botResponse = { q: "", a: response.data.response }
        setMessages(prevMessages => [...prevMessages, newMessage, botResponse])
      }
    } catch (error) {
      console.error("Error:", error)
    }

    setQuestion("")
  }

  // Handle voice input
  const handleVoiceInput = () => {
    if (!("SpeechRecognition" in window || "webkitSpeechRecognition" in window)) {
      alert("Speech Recognition is not supported in this browser.")
      return
    }

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)()
    recognition.lang = "en-US"
    recognition.interimResults = false
    recognition.maxAlternatives = 1

    recognition.onstart = () => {
      console.log("Voice recognition started.")
    }

    recognition.onresult = (event: any) => {
      const speechToText = event.results[0][0].transcript
      setQuestion(speechToText)
      handleTextInput() // Automatically trigger text input handling after speech input
    }

    recognition.onerror = (event: any) => {
      console.error("Speech recognition error", event)
    }

    recognition.onend = () => {
      console.log("Voice recognition ended.")
    }

    recognition.start()
  }

  return (
    <Box maxWidth="500px" mx="auto" p="4" border="1px" borderColor="gray.300" borderRadius="md">
      <VStack spacing="4" align="stretch">
        {/* Chat display */}
        <Box maxHeight="400px" overflowY="auto" border="1px" borderColor="gray.200" p="4" borderRadius="md">
          {messages.map((message, index) => (
            <Flex
              key={index}
              direction={message.q ? "row-reverse" : "row"}
              align="center"
              justify="flex-start"
              mb="3"
            >
              <Box
                p="2"
                borderRadius="md"
                bg={message.q ? "blue.500" : "green.500"}
                color="white"
                maxWidth="70%"
              >
                <Text>{message.q || message.a}</Text>
              </Box>
            </Flex>
          ))}
        </Box>

        {/* Text input */}
        <HStack spacing="4" align="center">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question"
            size="md"
            width="80%"
          />
          <Button onClick={handleTextInput} colorScheme="blue">Send</Button>
          <IconButton
            aria-label="Start voice input"
            icon={<FaMicrophone />}
            onClick={handleVoiceInput}
            colorScheme="green"
          />
        </HStack>
      </VStack>
    </Box>
  )
}

export default Chat
