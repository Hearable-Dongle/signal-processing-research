type ZipEntry = {
  path: string;
  bytes: Uint8Array;
};

const textEncoder = new TextEncoder();

function writeUint16LE(target: Uint8Array, offset: number, value: number): void {
  target[offset] = value & 0xff;
  target[offset + 1] = (value >>> 8) & 0xff;
}

function writeUint32LE(target: Uint8Array, offset: number, value: number): void {
  target[offset] = value & 0xff;
  target[offset + 1] = (value >>> 8) & 0xff;
  target[offset + 2] = (value >>> 16) & 0xff;
  target[offset + 3] = (value >>> 24) & 0xff;
}

let crcTable: Uint32Array | null = null;

function getCrcTable(): Uint32Array {
  if (crcTable) {
    return crcTable;
  }
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  crcTable = table;
  return table;
}

function crc32(data: Uint8Array): number {
  const table = getCrcTable();
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i += 1) {
    crc = table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function textFile(path: string, text: string): ZipEntry {
  return { path, bytes: textEncoder.encode(text) };
}

export function createZipBlob(entries: ZipEntry[]): Blob {
  let offset = 0;
  const localParts: Uint8Array[] = [];
  const centralParts: Uint8Array[] = [];

  for (const entry of entries) {
    const name = textEncoder.encode(entry.path);
    const content = entry.bytes;
    const crc = crc32(content);

    const localHeader = new Uint8Array(30 + name.length);
    writeUint32LE(localHeader, 0, 0x04034b50);
    writeUint16LE(localHeader, 4, 20);
    writeUint16LE(localHeader, 6, 0);
    writeUint16LE(localHeader, 8, 0);
    writeUint16LE(localHeader, 10, 0);
    writeUint16LE(localHeader, 12, 0);
    writeUint32LE(localHeader, 14, crc);
    writeUint32LE(localHeader, 18, content.length);
    writeUint32LE(localHeader, 22, content.length);
    writeUint16LE(localHeader, 26, name.length);
    localHeader.set(name, 30);

    const centralHeader = new Uint8Array(46 + name.length);
    writeUint32LE(centralHeader, 0, 0x02014b50);
    writeUint16LE(centralHeader, 4, 20);
    writeUint16LE(centralHeader, 6, 20);
    writeUint16LE(centralHeader, 8, 0);
    writeUint16LE(centralHeader, 10, 0);
    writeUint16LE(centralHeader, 12, 0);
    writeUint16LE(centralHeader, 14, 0);
    writeUint32LE(centralHeader, 16, crc);
    writeUint32LE(centralHeader, 20, content.length);
    writeUint32LE(centralHeader, 24, content.length);
    writeUint16LE(centralHeader, 28, name.length);
    writeUint16LE(centralHeader, 30, 0);
    writeUint16LE(centralHeader, 32, 0);
    writeUint16LE(centralHeader, 34, 0);
    writeUint16LE(centralHeader, 36, 0);
    writeUint32LE(centralHeader, 38, 0);
    writeUint32LE(centralHeader, 42, offset);
    centralHeader.set(name, 46);

    localParts.push(localHeader, content);
    centralParts.push(centralHeader);
    offset += localHeader.length + content.length;
  }

  const centralSize = centralParts.reduce((sum, part) => sum + part.length, 0);
  const end = new Uint8Array(22);
  writeUint32LE(end, 0, 0x06054b50);
  writeUint16LE(end, 4, 0);
  writeUint16LE(end, 6, 0);
  writeUint16LE(end, 8, entries.length);
  writeUint16LE(end, 10, entries.length);
  writeUint32LE(end, 12, centralSize);
  writeUint32LE(end, 16, offset);
  writeUint16LE(end, 20, 0);

  const parts = [...localParts, ...centralParts, end].map(
    (part) => part.buffer.slice(part.byteOffset, part.byteOffset + part.byteLength) as ArrayBuffer
  );
  return new Blob(parts, { type: "application/zip" });
}
