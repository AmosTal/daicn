import { Box, Heading, Table, Thead, Tbody, Tr, Th, Td, Button } from '@chakra-ui/react';
import { useQuery } from 'react-query';
import { useState } from 'react';
import { useWeb3 } from '../context/Web3Context';

interface Task {
  id: string;
  client: string;
  computeUnits: number;
  reward: string;
  status: string;
}

export default function TaskList() {
  const { account } = useWeb3();
  const [tasks, setTasks] = useState<Task[]>([]);

  const { isLoading } = useQuery('tasks', async () => {
    // TODO: Implement API call to fetch tasks
    const response = await fetch('http://localhost:8000/tasks');
    const data = await response.json();
    setTasks(data);
  });

  return (
    <Box>
      <Heading size="lg" mb={4}>Active Tasks</Heading>
      <Table variant="simple">
        <Thead>
          <Tr>
            <Th>Task ID</Th>
            <Th>Client</Th>
            <Th>Compute Units</Th>
            <Th>Reward</Th>
            <Th>Status</Th>
            <Th>Action</Th>
          </Tr>
        </Thead>
        <Tbody>
          {tasks.map((task) => (
            <Tr key={task.id}>
              <Td>{task.id}</Td>
              <Td>{`${task.client.slice(0, 6)}...${task.client.slice(-4)}`}</Td>
              <Td>{task.computeUnits}</Td>
              <Td>{task.reward} DAICN</Td>
              <Td>{task.status}</Td>
              <Td>
                <Button
                  size="sm"
                  isDisabled={task.client === account}
                  onClick={() => {
                    // TODO: Implement task acceptance logic
                  }}
                >
                  Accept Task
                </Button>
              </Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
    </Box>
  );
}
