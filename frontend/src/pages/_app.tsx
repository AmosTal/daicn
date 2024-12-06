import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import type { AppProps } from 'next/app';
import { Web3Provider } from '../context/Web3Context';
import theme from '../theme';

const queryClient = new QueryClient();

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <QueryClientProvider client={queryClient}>
      <ChakraProvider theme={theme}>
        <Web3Provider>
          <Component {...pageProps} />
        </Web3Provider>
      </ChakraProvider>
    </QueryClientProvider>
  );
}

export default MyApp;
