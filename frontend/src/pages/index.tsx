import { Box, Container, Heading, Text, Button, VStack, HStack, useColorModeValue } from '@chakra-ui/react';
import { useWeb3 } from '../context/Web3Context';
import TaskList from '../components/TaskList';
import ProviderStats from '../components/ProviderStats';

export default function Home() {
  const { account, connectWallet, isConnecting } = useWeb3();
  const bgColor = useColorModeValue('gray.50', 'gray.900');

  return (
    <Box minH="100vh" bg={bgColor}>
      <Container maxW="container.xl" py={8}>
        <VStack spacing={8} align="stretch">
          <HStack justify="space-between">
            <VStack align="start">
              <Heading size="2xl">DAICN</Heading>
              <Text fontSize="lg" color="gray.500">
                Decentralized AI Computation Network
              </Text>
            </VStack>
            {!account ? (
              <Button
                size="lg"
                onClick={connectWallet}
                isLoading={isConnecting}
                loadingText="Connecting..."
              >
                Connect Wallet
              </Button>
            ) : (
              <Text>Connected: {`${account.slice(0, 6)}...${account.slice(-4)}`}</Text>
            )}
          </HStack>

          {account && (
            <>
              <ProviderStats />
              <TaskList />
            </>
          )}
        </VStack>
      </Container>
    </Box>
  );
}
