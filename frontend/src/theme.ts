import { extendTheme } from '@chakra-ui/react';

const theme = extendTheme({
  config: {
    initialColorMode: 'dark',
    useSystemColorMode: false,
  },
  colors: {
    brand: {
      50: '#E5F4FF',
      100: '#B8E1FF',
      200: '#8ACEFF',
      300: '#5CBBFF',
      400: '#2EA8FF',
      500: '#0095FF',
      600: '#0077CC',
      700: '#005999',
      800: '#003B66',
      900: '#001D33',
    },
  },
  styles: {
    global: {
      body: {
        bg: 'gray.900',
        color: 'white',
      },
    },
  },
  components: {
    Button: {
      defaultProps: {
        colorScheme: 'brand',
      },
    },
  },
});

export default theme;
