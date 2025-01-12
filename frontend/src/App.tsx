import React from 'react'
import { ChakraProvider } from '@chakra-ui/react'
import { theme } from './styles/globalStyles'
import Chat from './components/chat'

const App: React.FC = () => {
  return (
    <ChakraProvider theme={theme}>
      <Chat />
    </ChakraProvider>
  )
}

export default App

