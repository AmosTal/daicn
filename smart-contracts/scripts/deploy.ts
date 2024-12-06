import { ethers } from "hardhat";

async function main() {
  // Deploy DAICNToken
  console.log("Deploying DAICNToken...");
  const DAICNToken = await ethers.getContractFactory("DAICNToken");
  const token = await DAICNToken.deploy();
  await token.deployed();
  console.log("DAICNToken deployed to:", token.address);

  // Deploy ComputeMarketplace
  console.log("Deploying ComputeMarketplace...");
  const ComputeMarketplace = await ethers.getContractFactory("ComputeMarketplace");
  const marketplace = await ComputeMarketplace.deploy(token.address);
  await marketplace.deployed();
  console.log("ComputeMarketplace deployed to:", marketplace.address);

  // Transfer some initial tokens to the marketplace
  const initialMarketplaceTokens = ethers.utils.parseEther("1000000");
  await token.transfer(marketplace.address, initialMarketplaceTokens);
  console.log("Transferred initial tokens to marketplace");

  // Verify contracts on Etherscan
  if (process.env.ETHERSCAN_API_KEY) {
    console.log("Verifying contracts on Etherscan...");
    await verify(token.address, []);
    await verify(marketplace.address, [token.address]);
  }
}

async function verify(contractAddress: string, args: any[]) {
  try {
    await hre.run("verify:verify", {
      address: contractAddress,
      constructorArguments: args,
    });
  } catch (e) {
    console.log("Verification failed:", e);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
