import { Box, SimpleGrid, Stat, StatLabel, StatNumber, StatHelpText } from '@chakra-ui/react';
import { useQuery } from 'react-query';
import { useWeb3 } from '../context/Web3Context';

export default function ProviderStats() {
  const { account } = useWeb3();

  const { data: stats, isLoading } = useQuery(['providerStats', account], async () => {
    // TODO: Implement API call to fetch provider stats
    return {
      computePower: '1000 FLOPS',
      activeJobs: 5,
      totalEarned: '500 DAICN',
      reputation: 98,
    };
  });

  return (
    <Box p={5} shadow="xl" borderRadius="lg" bg="gray.800">
      <SimpleGrid columns={{ base: 2, md: 4 }} spacing={5}>
        <Stat>
          <StatLabel>Compute Power</StatLabel>
          <StatNumber>{stats?.computePower || '-'}</StatNumber>
          <StatHelpText>Available Resources</StatHelpText>
        </Stat>

        <Stat>
          <StatLabel>Active Jobs</StatLabel>
          <StatNumber>{stats?.activeJobs || 0}</StatNumber>
          <StatHelpText>Currently Processing</StatHelpText>
        </Stat>

        <Stat>
          <StatLabel>Total Earned</StatLabel>
          <StatNumber>{stats?.totalEarned || '0 DAICN'}</StatNumber>
          <StatHelpText>Lifetime Earnings</StatHelpText>
        </Stat>

        <Stat>
          <StatLabel>Reputation</StatLabel>
          <StatNumber>{stats?.reputation || 0}%</StatNumber>
          <StatHelpText>Provider Rating</StatHelpText>
        </Stat>
      </SimpleGrid>
    </Box>
  );
}
