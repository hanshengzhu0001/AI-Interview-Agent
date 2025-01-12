// src/types/speechRecognition.d.ts

declare global {
    interface SpeechRecognitionEvent {
      results: SpeechRecognitionResultList;
    }
  
    interface SpeechRecognition extends EventTarget {
      lang: string;
      interimResults: boolean;
      maxAlternatives: number;
      continuous: boolean;
      start(): void;
      stop(): void;
      abort(): void;
  
      onstart?: () => void;
      onend?: () => void;
      onresult?: (event: SpeechRecognitionEvent) => void;
      onerror?: (event: Event) => void;
      onnomatch?: () => void;
      onspeechstart?: () => void;
      onspeechend?: () => void;
      onsoundstart?: () => void;
      onsoundend?: () => void;
      onnoise?: () => void;
    }
  
    interface Window {
      SpeechRecognition: typeof SpeechRecognition;
      webkitSpeechRecognition: typeof SpeechRecognition;
    }
  }
  
  export {};
  
  