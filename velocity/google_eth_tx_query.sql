SELECT
  DATE(block_timestamp) as TxDate,
  token_transfers.*
FROM
  `bigquery-public-data.ethereum_blockchain.token_transfers` AS token_transfers
WHERE
  token_transfers.token_address = '0x89d24a6b4ccb1b6faa2625fe562bdd9a23260359'
ORDER BY
  block_number


# GUSD: 0x056fd409e1d7a124bd7017459dfea2f387b6d5cd
# PAX: 0x8e870d67f660d95d5be530380d0ec0bd388289e1
# TUSD: 0x8dd5fbce2f6a956c3022ba3663759011dd51e73e
# USDC: 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48