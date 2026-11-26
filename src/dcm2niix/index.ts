// DICOM drops to NIfTI with @niivue/dcm2niix (adapted from EdgeReg, BSD-2-Clause).
import { Dcm2niix } from '@niivue/dcm2niix'

// dcm2niix groups slices into series by this non-standard path (File.webkitRelativePath is read-only).
type FileWithRelativePath = File & { _webkitRelativePath?: string }

// The package ignores worker errors before its `ready` message; bound init so a failure still terminates.
const INIT_TIMEOUT_MS = 30_000

/** DICOM files to the NIfTI images dcm2niix makes of them. */
// Names from protocol and series only: dcm2niix's default adds the folder (often the patient) and date.
export async function runDcm2niix(files: File[]): Promise<File[]> {
  if (!files.length) return []
  const dcm2niix = new Dcm2niix()
  let timer: number | undefined
  try {
    await Promise.race([
      dcm2niix.init(),
      new Promise<never>((_, reject) => {
        timer = globalThis.setTimeout(() => reject(new Error('dcm2niix worker initialization timed out')), INIT_TIMEOUT_MS)
      }),
    ])
    const result = (await dcm2niix.input(files).f('%p_%s').run()) as File[]
    return result.filter((f) => /\.nii(\.gz)?$/i.test(f.name))
  } finally {
    globalThis.clearTimeout(timer)
    dcm2niix.worker?.terminate()
  }
}

/** A drop's files, recursing into folders and recording each file's relative path. */
export async function traverseDataTransferItems(items: DataTransferItemList): Promise<File[]> {
  const files: File[] = []
  const entries = Array.from(items, (item) => item.webkitGetAsEntry()).filter((e): e is FileSystemEntry => !!e)
  await Promise.all(entries.map((entry) => walk(entry, '', files)))
  return files
}

function walk(entry: FileSystemEntry, path: string, out: File[]): Promise<void> {
  return new Promise((resolve, reject) => {
    if (entry.isFile) {
      ;(entry as FileSystemFileEntry).file((file) => {
        ;(file as FileWithRelativePath)._webkitRelativePath = path + file.name
        out.push(file)
        resolve()
      }, reject)
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader()
      const read = () =>
        reader.readEntries((batch) => {
          if (!batch.length) return resolve()
          Promise.all(batch.map((child) => walk(child, `${path}${entry.name}/`, out))).then(read, reject)
        }, reject)
      read()
    } else {
      resolve()
    }
  })
}
