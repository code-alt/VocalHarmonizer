#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

//==============================================================================
/**
*/
class HarmonizeAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    HarmonizeAudioProcessorEditor (HarmonizeAudioProcessor&);
    ~HarmonizeAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    HarmonizeAudioProcessor& audioProcessor;

    juce::Slider dryLevelSlider;
    juce::Slider harmonyLevelSlider;
    juce::Slider stereoSpreadSlider;
    juce::Label dryLevelLabel;
    juce::Label harmonyLevelLabel;
    juce::Label stereoSpreadLabel;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> dryLevelAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> harmonyLevelAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> stereoSpreadAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (HarmonizeAudioProcessorEditor)
};
