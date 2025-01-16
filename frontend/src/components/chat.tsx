import React, { useState } from 'react';
import { Input, Button, Box, VStack, Text, IconButton, Flex, HStack } from '@chakra-ui/react';
import axios from 'axios';
import { FaMicrophone, FaPlay } from 'react-icons/fa';

const Chat: React.FC = () => {
  const [question, setQuestion] = useState<string>('');
  const [messages, setMessages] = useState<{ q: string; a: string; audio?: string }[]>([]);
  const [recording, setRecording] = useState<boolean>(false);

  const handleTextInput = async () => {
    if (!question.trim()) return;

    const userMessage = { q: question, a: '' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    try {
      const response = await axios.post(
        'http://127.0.0.1:5000/rasa',
        { message: question },
        { headers: { 'Content-Type': 'application/json' } }
      );

      const { response: botResponse, audio_path } = response.data;

      setMessages((prevMessages) => [
        ...prevMessages,
        { q: '', a: botResponse, audio: audio_path },
      ]);
    } catch (error) {
      console.error('Error processing text input:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { q: '', a: 'Error occurred while processing your input.' },
      ]);
    }
    setQuestion('');
  };

  const handleVoiceInput = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('voice', blob);

        try {
          const response = await axios.post('http://127.0.0.1:5000/rasa', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });

          const { response: botResponse, audio_path, transcription } = response.data;

          const userMessage = transcription
            ? { q: transcription, a: '' }
            : { q: 'Voice Input', a: 'Sorry, I could not understand the audio.' };

          setMessages((prevMessages) => [
            ...prevMessages,
            userMessage,
            { q: '', a: botResponse || 'Error occurred while processing your input.', audio: audio_path },
          ]);
        } catch (error) {
          console.error('Error during voice processing:', error);
          setMessages((prevMessages) => [
            ...prevMessages,
            { q: 'Voice Input', a: 'Error occurred while processing your input.' },
          ]);
        }
      };

      recorder.start();
      setRecording(true);
      setTimeout(() => {
        recorder.stop();
        setRecording(false);
      }, 5000);
    } catch (error) {
      console.error('Error accessing the microphone:', error);
      alert('Unable to access the microphone. Please check your browser settings.');
    }
  };

  const handlePlayAudio = async (audioPath: string) => {
    if (!audioPath) return;

    const audio = new Audio(`http://127.0.0.1:5000${audioPath}`);
    audio.play();
  };

  const handleMessageClick = (audioPath?: string) => {
    if (audioPath) handlePlayAudio(audioPath);
  };

  return (
    <Box maxWidth="500px" mx="auto" p="4" border="1px" borderColor="gray.300" borderRadius="md">
      <VStack spacing="4" align="stretch">
        {/* Message Display */}
        <Box
          maxHeight="400px"
          overflowY="auto"
          border="1px"
          borderColor="gray.200"
          p="4"
          borderRadius="md"
        >
          {messages.map((message, index) => (
            <Flex
              key={index}
              direction={message.q ? 'row-reverse' : 'row'}
              align="center"
              mb="3"
              onClick={() => handleMessageClick(message.audio)}
              cursor={message.audio ? 'pointer' : 'default'}
            >
              <Box
                p="2"
                borderRadius="md"
                bg={message.q ? 'blue.500' : 'green.500'}
                color="white"
                maxWidth="70%"
              >
                <Text>{message.q || message.a}</Text>
              </Box>
              {message.audio && (
                <IconButton
                  aria-label="Play audio"
                  icon={<FaPlay />}
                  onClick={(e) => {
                    e.stopPropagation(); // Prevent triggering the click on parent
                    handlePlayAudio(message.audio!);
                  }}
                  colorScheme="teal"
                  ml="2"
                />
              )}
            </Flex>
          ))}
        </Box>

        {/* Input and Actions */}
        <HStack spacing="4">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question"
          />
          <Button onClick={handleTextInput} colorScheme="blue">
            Send
          </Button>
          <IconButton
            aria-label="Voice input"
            icon={<FaMicrophone />}
            onClick={handleVoiceInput}
            colorScheme={recording ? 'red' : 'green'}
          />
        </HStack>
      </VStack>
    </Box>
  );
};

export default Chat;