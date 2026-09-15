# Distributed Systems Fundamentals

## What Is a Distributed System?

A distributed system is a collection of independent computers that appear to users as a single coherent system. The nodes communicate by passing messages over a network. Examples include databases like Cassandra and CockroachDB, message brokers like Apache Kafka, and globally distributed applications like Google Search and Amazon.

Distributed systems offer benefits such as horizontal scalability, fault tolerance, and geographic distribution, but introduce fundamental challenges around consistency, availability, and partition tolerance.

---

## The CAP Theorem

The CAP theorem, proved by Eric Brewer in 2000, states that a distributed data store can guarantee at most two of the following three properties simultaneously:

**Consistency (C)**: Every read receives the most recent write or an error. All nodes see the same data at the same time.

**Availability (A)**: Every request receives a (non-error) response, but it might not be the most recent write.

**Partition Tolerance (P)**: The system continues to operate even when network partitions (message loss or delay) occur between nodes.

Because network partitions are unavoidable in practice, real distributed systems must choose between CP (sacrifice availability during partitions) and AP (sacrifice consistency during partitions).

- **CP examples**: HBase, Zookeeper, etcd — return errors rather than stale data.
- **AP examples**: Cassandra, CouchDB, DynamoDB (in default configuration) — return potentially stale data rather than errors.

---

## Consistency Models

Not all consistency is binary. A spectrum of consistency models trades off between strong guarantees and performance:

**Strong consistency (linearisability)**: Operations appear to execute atomically at some point between their invocation and response. The system behaves as if there is a single, current copy of the data. This is the strongest guarantee but requires coordination between nodes.

**Sequential consistency**: All nodes see operations in the same order, but that order need not match real time. Stronger than causal but weaker than linearisability.

**Causal consistency**: Operations that are causally related are seen by all nodes in the same order. Causally unrelated operations may be seen in different orders. Used in systems like MongoDB and some versions of DynamoDB.

**Eventual consistency**: If no new updates are made, all replicas will eventually converge to the same value. No guarantee on when. Highest availability and performance. Used in DNS, Cassandra, and Amazon S3.

**Read-your-writes consistency**: A user always sees their own writes, even if other users may see stale data. A common practical requirement.

---

## Replication

Replication copies data to multiple nodes to increase availability and read throughput.

### Leader–Follower (Primary–Replica) Replication
One node (the leader) handles all writes. Changes are propagated to follower nodes, which serve read queries. If the leader fails, a follower is promoted. Used in MySQL, PostgreSQL, and Redis.

**Synchronous replication**: the leader waits for acknowledgement from at least one follower before confirming the write. Guarantees no data loss on leader failure but increases write latency.

**Asynchronous replication**: the leader confirms the write without waiting. Lower latency but followers can fall behind; data loss is possible if the leader fails before propagating.

### Multi-Leader Replication
Multiple nodes can accept writes. Useful for multi-datacenter deployments. Write conflicts must be resolved (last-write-wins, custom merge functions, or CRDTs).

### Leaderless Replication
Any node can accept writes. Reads and writes are sent to multiple nodes simultaneously; quorums determine success. Cassandra and DynamoDB use leaderless replication with quorum reads/writes.

A quorum write requires W nodes to acknowledge; a quorum read contacts R nodes. For strong consistency, W + R > N where N is the replication factor.

---

## Consensus Algorithms

Consensus is the problem of getting a set of nodes to agree on a single value even when some nodes may fail.

### Paxos
Paxos, proposed by Leslie Lamport, is the foundational consensus algorithm. It operates in two phases: Prepare (a proposer asks acceptors to promise not to accept older proposals) and Accept (the proposer sends its value; acceptors accept if they haven't promised otherwise). Multi-Paxos extends this for a sequence of values (a replicated log).

### Raft
Raft was designed to be more understandable than Paxos. It decomposes consensus into leader election, log replication, and safety. A leader is elected by receiving votes from a majority. All writes go through the leader, which appends entries to its log and replicates them to followers. A log entry is committed once a majority acknowledges it. etcd, CockroachDB, and TiKV use Raft.

---

## Distributed Transactions

Coordinating a transaction across multiple nodes requires special protocols.

### Two-Phase Commit (2PC)
A coordinator sends a Prepare request to all participants. Each participant votes Yes (and locks resources) or No. If all vote Yes, the coordinator sends Commit; otherwise it sends Abort.

2PC is blocking: if the coordinator crashes after participants have voted Yes but before sending Commit/Abort, participants are stuck holding locks indefinitely.

### Saga Pattern
For long-lived transactions across microservices, the Saga pattern decomposes the transaction into a sequence of local transactions, each with a compensating transaction that can undo it. If any step fails, compensating transactions are executed in reverse order. This avoids distributed locks but requires idempotent operations and careful design.

---

## Message Queues and Event Streaming

### Apache Kafka
Kafka is a distributed log. Producers append records to topics, which are divided into partitions replicated across brokers. Consumers read from partitions at their own pace using offsets. Kafka guarantees at-least-once delivery by default; exactly-once semantics are available via transactions.

Kafka retains messages for a configurable period (default 7 days), enabling replay. This makes it suitable for event sourcing, stream processing with Apache Flink or Kafka Streams, and decoupling microservices.

### Message Ordering
Kafka guarantees ordering within a partition, not across partitions. Producers assign records to partitions by key (e.g., order ID); all records with the same key go to the same partition, ensuring order per entity.

---

## Common Failure Modes

| Failure | Description | Mitigation |
|---|---|---|
| Network partition | Nodes cannot communicate | Design for AP or CP as appropriate |
| Node crash | A node stops responding | Replication + leader election |
| Slow node (grey failure) | A node is alive but slow | Timeouts, hedged requests |
| Split brain | Two leaders believe they are authoritative | Fencing tokens, STONITH |
| Clock skew | Clocks on different nodes diverge | Hybrid logical clocks, TrueTime (Google Spanner) |
| Thundering herd | Many clients retry simultaneously after failure | Exponential backoff with jitter |

---

## The Fallacies of Distributed Computing

Peter Deutsch's famous list of incorrect assumptions developers often make:

1. The network is reliable.
2. Latency is zero.
3. Bandwidth is infinite.
4. The network is secure.
5. Topology does not change.
6. There is one administrator.
7. Transport cost is zero.
8. The network is homogeneous.

Designing robust distributed systems requires accounting for all eight fallacies.
