/*
  
  VocalHarmonizer
    
*/

#pragma once

#include <JuceHeader.h>
#include <atomic>
#include "HarmonizerDSP.h"

//==============================================================================
class HarmonizeAudioProcessor : public juce::AudioProcessor
{
public:
    //==============================================================================
    HarmonizeAudioProcessor();
    ~HarmonizeAudioProcessor() override;

    using ParameterState = juce::AudioProcessorValueTreeState;
    ParameterState& getParameters() noexcept { return parameters; }

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void reset() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
#endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlockBypassed (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

private:
    //==============================================================================
    static ParameterState::ParameterLayout createParameterLayout();

    void processInternal (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages, bool bypassed);
    void handleMidiMessage (const juce::MidiMessage& message);

    //==============================================================================
    ParameterState parameters;
    std::atomic<float>* dryLevelParameter = nullptr;
    std::atomic<float>* harmonyLevelParameter = nullptr;
    std::atomic<float>* stereoSpreadParameter = nullptr;

    harmonize::HarmonizerCore core;
    juce::SmoothedValue<float> bypassGain { 1.0f };
    std::atomic<bool> resetPending { false };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HarmonizeAudioProcessor)
};
